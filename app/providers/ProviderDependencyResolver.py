"""Rilevamento conservativo, senza eseguire sorgenti o generatori ReDataset.

Riconosce API cloud e wrapper SeBS originali; la scelta del backend AWS
per i wrapper multi-cloud e' una *policy esplicita*, non un'inferenza che
il codice utilizzi necessariamente AWS. Non modifica i sorgenti.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


class ProviderDependencyError(RuntimeError):
    """Provider noto ma non eseguibile, oppure individuazione ambigua."""


@dataclass(frozen=True)
class ProviderDependency:
    provider: str | None
    services: tuple[str, ...]
    status: str                      # none | supported | unsupported
    detection: str
    evidence: tuple[str, ...]
    library_requirements: tuple[str, ...] = ()
    backend_policy: str | None = None

    def as_dict(self) -> dict:
        return {
            "provider": self.provider, "services": list(self.services),
            "status": self.status, "detection": self.detection,
            "evidence": list(self.evidence),
            "library_requirements": list(self.library_requirements),
            "backend_policy": self.backend_policy,
        }

    def announcement(self, task_id: str) -> str:
        if self.provider is None:
            if self.status != "none":
                return (f"[Dependencies] {task_id}: provider=ambiguo "
                        f"| supporto={self.status} | motivo={self.detection}")
            packages = ", ".join(self.library_requirements[:4]) or "nessuna dichiarata"
            return f"[Dependencies] {task_id}: provider=nessuno | librerie={packages}"
        services = ",".join(self.services) or "servizio non determinato"
        suffix = f" | scelta={self.backend_policy}" if self.backend_policy else ""
        return (f"[Dependencies] {task_id}: provider={self.provider.upper()} "
                f"| servizi={services} | supporto={self.status}{suffix}")


# Alias di librerie Python esplicitamente associate ad API cloud.
_CLOUD_REQUIREMENTS = {
    "boto3": ("aws", "unknown"),
    "botocore": ("aws", "unknown"),
    "azure-storage-blob": ("azure", "blob"),
    "azure-cosmos": ("azure", "cosmos"),
    "google-cloud-storage": ("gcp", "storage"),
    "google-cloud-datastore": ("gcp", "datastore"),
    "oci": ("oci", "unknown"),
}


def _requirement_names(text: str | None) -> tuple[str, ...]:
    """Solo nomi: niente esecuzione, resolver pip o interpretazione di URL."""
    names: set[str] = set()
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        found = re.match(r"([A-Za-z][A-Za-z0-9_.-]*)\s*(?:[<>=!~;\[].*)?$", line)
        if found:
            names.add(found.group(1).lower().replace("_", "-"))
    return tuple(sorted(names))


def _imports(tree: ast.AST) -> tuple[set[str], set[str]]:
    imports: set[str] = set()
    relative: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.update(alias.name for alias in node.names)
                if node.module:
                    relative.add(node.module.split(".", 1)[0])
            else:
                if node.module:
                    imports.add(node.module)
                    if node.module == "google.cloud":
                        imports.update("google.cloud." + alias.name for alias in node.names)
                    elif node.module in ("azure.storage", "azure"):
                        imports.update(node.module + "." + alias.name for alias in node.names)
    return imports, relative


def _aws_services(tree: ast.AST) -> set[str]:
    services: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (node.func.attr in ("client", "resource") and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in ("boto3", "session") and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                services.add(arg.value.lower())
    return services


class ProviderDependencyResolver:
    """Classifica una FunctionSource Python, indipendentemente dal dataset.

    Per SeBS la libreria relativa `storage`/`nosql` e' un provider soltanto
    se esiste il wrapper ufficiale nel corpus. Il backend multi-cloud viene
    selezionato in modo riproducibile: AWS per la prima implementazione.
    Per import cloud diretti rileva il provider ma non finge che il relativo
    esecutore dei test sia gia' disponibile.
    """

    def __init__(self, repository) -> None:
        self.repository = repository

    def resolve(self, source) -> ProviderDependency:
        candidate = source.candidate
        if candidate.language != "python":
            raise ProviderDependencyError("Il resolver accetta soltanto sorgenti Python")
        try:
            tree = ast.parse(source.code)
        except SyntaxError as exc:
            raise ProviderDependencyError("Impossibile analizzare gli import Python") from exc
        imports, relative = _imports(tree)
        requirements = _requirement_names(source.python_requirements_text)
        observed: dict[str, set[str]] = {}
        evidence: set[str] = set()
        for package, (provider, service) in _CLOUD_REQUIREMENTS.items():
            if package in requirements:
                observed.setdefault(provider, set()).add(service)
                evidence.add("requirement:" + package)
        for imported in imports:
            key = imported.lower()
            detected = None
            if key in ("boto3", "botocore") or key.startswith(("boto3.", "botocore.")):
                detected = ("aws", None)
            elif key.startswith("azure.storage.blob"):
                detected = ("azure", "blob")
            elif key.startswith("azure.cosmos"):
                detected = ("azure", "cosmos")
            elif key.startswith("google.cloud.storage"):
                detected = ("gcp", "storage")
            elif key.startswith("google.cloud.datastore"):
                detected = ("gcp", "datastore")
            elif key == "oci" or key.startswith("oci."):
                detected = ("oci", "unknown")
            if detected:
                provider, service = detected
                observed.setdefault(provider, set()).add(service or "unknown")
                evidence.add("import:" + imported)
        if "aws" in observed:
            found_services = _aws_services(tree)
            if found_services:
                observed["aws"].discard("unknown")
                observed["aws"].update(found_services)

        # Wrapper comuni di SeBS. Non trattare ogni modulo locale chiamato
        # 'storage' come una dipendenza da AWS.
        wrapper_services: set[str] = set()
        if candidate.source_id in ("sebs-master", "sebs-feature-workflows"):
            for module, service in (("storage", "s3"), ("nosql", "dynamodb")):
                if module not in relative:
                    continue
                wrapper = (self.repository.root / "sources" / candidate.source_id /
                           "benchmarks" / "wrappers" / "aws" / "python" / f"{module}.py")
                if wrapper.is_file() and wrapper.is_relative_to(self.repository.root):
                    wrapper_services.add(service)
                    evidence.add("sebs_original_wrapper:" + module)
                else:
                    raise ProviderDependencyError("Wrapper provider SeBS mancante: " + module)
        if wrapper_services:
            observed.setdefault("aws", set()).update(wrapper_services)
        if not observed:
            return ProviderDependency(None, (), "none", "static_python_analysis", (), requirements)
        if len(observed) > 1:
            names = ", ".join(sorted(observed))
            return ProviderDependency(None, (), "unsupported", "multiple_providers:" + names,
                                      tuple(sorted(evidence)), requirements)
        provider, services = next(iter(observed.items()))
        known = tuple(sorted(services))
        supported = (provider == "aws" and services == {"s3"} and
                     wrapper_services == {"s3"} and
                     candidate.source_id in ("sebs-master", "sebs-feature-workflows"))
        reason = "static_python_and_original_wrappers"
        if supported:
            # Il servizio puo' essere emulato, ma il generatore del benchmark
            # potrebbe avere effetti collaterali/asset non ancora supportati.
            # Una prova data-only, senza Docker o callback, evita di sprecare LLM.
            from app.validation.SeBSCases import SeBSCaseExtractor
            from app.validation.HumanevalCases import UnsupportedTestFormat
            from app.providers.ProviderHttpFixture import fixture_metadata
            try:
                extracted = SeBSCaseExtractor(generator_bindings={
                    "benchmarks_bucket": "spec2faas-detection-placeholder",
                    "input_buckets": [], "output_buckets": ["output"],
                }).prepare(source)
                original_case = extracted.cases[0].args[0]
                if (len(extracted.cases) != 1 or not source.tests or
                        fixture_metadata(source.tests[0].content, original_case) is None):
                    supported = False
                    reason = "provider_original_test_setup_not_supported"
            except (UnsupportedTestFormat, IndexError, TypeError, ValueError):
                supported = False
                reason = "provider_original_test_setup_not_supported"
            if not supported:
                from app.providers.ProviderAssetSetup import ProviderAssetSetup
                if ProviderAssetSetup.capability(source) is not None:
                    supported = ProviderAssetSetup.archive_path(self.repository).is_file()
                    reason = ("provider_catalogued_asset_setup" if supported else
                              "provider_asset_archive_missing")
        policy = "sebs_multicloud_prefer_aws" if wrapper_services else None
        return ProviderDependency(provider, known, "supported" if supported else "unsupported",
                                  reason, tuple(sorted(evidence)), requirements, policy)
