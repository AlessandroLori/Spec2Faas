import unittest
from types import SimpleNamespace

from app.Utils import total_tokens_from_response


# Test deterministici: nessuna richiesta a Ollama, MAF o Serverledge.
class TestPartialMAFUsage(unittest.TestCase):

    def test_usage_completo_continua_a_sommare_input_output(self):
        response = SimpleNamespace(usage_details={
            "input_token_count": 39,
            "output_token_count": 2,
            "total_token_count": 41,
        })
        self.assertEqual(total_tokens_from_response(response), 41)

    def test_solo_totale_ufficiale_dizionario(self):
        response = SimpleNamespace(usage_details={"total_token_count": 93})
        self.assertEqual(total_tokens_from_response(response), 93)

    def test_componente_mancante_ma_totale_disponibile(self):
        response = SimpleNamespace(usage_details={
            "input_token_count": 70,
            "total_token_count": 115,
        })
        self.assertEqual(total_tokens_from_response(response), 115)

    def test_totale_ufficiale_come_attributo(self):
        response = SimpleNamespace(usage_details=SimpleNamespace(
            total_token_count=28
        ))
        self.assertEqual(total_tokens_from_response(response), 28)

    def test_assenza_totale_rimane_errore(self):
        response = SimpleNamespace(usage_details={"input_token_count": 50})
        with self.assertRaisesRegex(RuntimeError, "available_fields") as cm:
            total_tokens_from_response(response)
        self.assertIn("input_token_count", str(cm.exception))

    def test_usage_vuoto_rimane_errore(self):
        response = SimpleNamespace(usage_details={})
        with self.assertRaisesRegex(RuntimeError, "available_fields"):
            total_tokens_from_response(response)

    def test_usage_none_rimane_errore(self):
        response = SimpleNamespace(usage_details=None)
        with self.assertRaisesRegex(RuntimeError, "usage_details=None"):
            total_tokens_from_response(response)

    def test_zero_e_un_valore_valido(self):
        response = SimpleNamespace(usage_details={
            "input_token_count": 0, "output_token_count": 0,
        })
        self.assertEqual(total_tokens_from_response(response), 0)


if __name__ == "__main__":
    unittest.main()
