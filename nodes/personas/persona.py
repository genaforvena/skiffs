class Persona:
    def __init__(self, name, model_name, persona_description):
        self.name = name
        self.model_name = model_name
        self.persona_description = persona_description

    def create_prompt_to_reply_to(self, last_reply, conversation_context):
        # TODO implement this
        return last_reply
