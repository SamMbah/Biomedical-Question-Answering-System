# UmlsClient.py

import requests

class UmlsClient:
    """
    Minimal UMLS REST client, updated to fetch definitions properly.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.auth_endpoint    = "https://utslogin.nlm.nih.gov"
        self.service_endpoint = "https://uts-ws.nlm.nih.gov"
        self.tgt = None  # Ticket‐Granting Ticket URL

    def authenticate(self):
        """
        Request a TGT. The response’s 'Location' header is the TGT URL.
        """
        data = { 'apikey': self.api_key }
        resp = requests.post(f"{self.auth_endpoint}/cas/v1/api-key", data=data)
        resp.raise_for_status()
        self.tgt = resp.headers['Location']
        return self.tgt

    def get_service_ticket(self, service: str = "http://umlsks.nlm.nih.gov"):
        """
        Given the TGT, request a short‐lived Service Ticket (ST) for 'service'.
        """
        if not self.tgt:
            raise RuntimeError("Call authenticate() first to obtain a TGT.")
        data = { 'service': service }
        resp = requests.post(self.tgt, data=data)
        resp.raise_for_status()
        return resp.text

    def _query(self, path: str, params: dict):
        """
        Generic GET to UMLS endpoint. Must include a valid ST in params['ticket'].
        """
        url = f"{self.service_endpoint}{path}"
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def find_cuis_for_text(self, text: str):
        """
        Use /rest/search/current to find candidate CUIs.
        """
        st = self.get_service_ticket()
        params = {
            'string': text,
            'ticket': st,
            'pageNumber': 1,
            'pageSize': 5
        }
        return self._query("/rest/search/current", params)

    def get_cui_definitions(self, cui: str):
        """
        Fetch the actual definitions by calling /rest/content/current/CUI/{cui}/definitions.
        """
        st = self.get_service_ticket()
        path = f"/rest/content/current/CUI/{cui}/definitions"
        params = { 'ticket': st }
        return self._query(path, params)
