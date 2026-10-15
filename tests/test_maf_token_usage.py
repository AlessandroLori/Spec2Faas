
import unittest
from types import SimpleNamespace

from app.Utils import total_tokens_from_response
from app.agents.Assistant import Assistant
from app.agents.FaaSDeployer import FaasDeployer
from app.agents.coding_agents.Coder import Coder
from app.agents.coding_agents.Debugger import Debugger
from app.agents.coding_agents.EntryPoint import EntryPoint
from app.agents.coding_agents.TestDesigner import TestDesigner
from app.agents.coding_agents.TestExecutor import TestExecutor


# Questi test non richiedono Ollama, Docker o Serverledge.
# Verificano il formato dei dati osservato con Ollama reale.


class TestMAFTokenUsage(unittest.TestCase):

    def test_dizionario_maf_reale(self):
        # Formato restituito dal provider Ollama.
        response = SimpleNamespace(
            usage_details={
                "input_token_count": 33,
                "output_token_count": 2,
                "total_token_count": 35,
            }
        )

        self.assertEqual(
            total_tokens_from_response(response),
            35,
        )

    def test_oggetto_compatibile(self):
        # Formato utilizzato nei test precedenti.
        response = SimpleNamespace(
            usage_details=SimpleNamespace(
                input_token_count=33,
                output_token_count=2,
                total_token_count=35,
            )
        )

        self.assertEqual(
            total_tokens_from_response(response),
            35,
        )

    def test_zero_token(self):
        # Il valore 0 è valido e non deve essere
        # confuso con un campo mancante.
        response = SimpleNamespace(
            usage_details={
                "input_token_count": 0,
                "output_token_count": 0,
            }
        )

        self.assertEqual(
            total_tokens_from_response(response),
            0,
        )

    def test_metriche_assenti(self):
        response = SimpleNamespace(
            usage_details=None
        )

        with self.assertRaises(RuntimeError):
            total_tokens_from_response(response)

    def test_campo_mancante(self):
        response = SimpleNamespace(
            usage_details={
                "input_token_count": 33,
            }
        )

        with self.assertRaises(RuntimeError):
            total_tokens_from_response(response)

    def test_tipo_eccezione_personalizzato(self):
        # Il Debugger utilizzava ValueError:
        # verifichiamo che sia ancora possibile.
        response = SimpleNamespace(
            usage_details=None
        )

        with self.assertRaises(ValueError):
            total_tokens_from_response(
                response,
                error_message="Missing LLM token usage details",
                error_type=ValueError,
            )

    def test_tutti_e_sette_gli_agenti(self):
        # Verifica dei metodi realmente utilizzati
        # dagli agenti migrati.
        methods = {
            "EntryPoint": EntryPoint._total_tokens,
            "Coder": Coder._total_tokens,
            "TestDesigner": TestDesigner._total_tokens,
            "TestExecutor": TestExecutor._total_tokens,
            "Debugger": Debugger._total_tokens,
            "Assistant": Assistant._tokens,
            "FaasDeployer": FaasDeployer._total_tokens,
        }

        formats = {
            "dict": {
                "input_token_count": 33,
                "output_token_count": 2,
                "total_token_count": 35,
            },
            "object": SimpleNamespace(
                input_token_count=33,
                output_token_count=2,
                total_token_count=35,
            ),
        }

        # Ogni agente deve accettare entrambi i formati.
        for agent_name, method in methods.items():
            for format_name, usage in formats.items():

                with self.subTest(
                    agent=agent_name,
                    format=format_name,
                ):
                    response = SimpleNamespace(
                        usage_details=usage
                    )

                    self.assertEqual(
                        method(response),
                        35,
                    )


if __name__ == "__main__":
    unittest.main()
