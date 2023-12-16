class Persona:
    def __init__(self, name, model_name, persona_description):
        self.name = name
        self.model_name = model_name
        self.persona_description = persona_description

    def add_reply_to(self, conversation_history: str):
        # TODO implement this
        pass

    def _get_last_reply(self, conversation_history: str):
        # Method stub for now
        pass

    def _extract_last_message(self, conversation_history: str):
        pass

    def _extract_conversation_context(self, conversation_history: str):
        pass
