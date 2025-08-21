class InstructionTuningPair:
    def __init__(self, language: str, prompt: str, response: str):
        self.language = language
        self.prompt = prompt
        self.response = response

    def __str__(self):
        return f"Language: {self.language}\nPrompt: {self.prompt}\nResponse: {self.response}"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    # let's do some Zulu

    language = "Zulu"

    primary_language = "Uqhube wathi inkantolo akufanele ivume ukuthi yenze iphutha ngoba abukho ubufakazi bobudedengu ngokwezimali. Ngaphezu kwalokho, uthe abukho ubufakazi bokuthi wake wamosha impahla ehlangene nesoka lakhe, njengokwesinqumo esakhishwa yiMantshi ngesimangalo sowayengumyeni wakhe."

    translated_to_en_by_machine_translation = "He continued saying that the court should not admit that it made a mistake because there was no evidence of financial negligence. In addition, he said there was no evidence that he ever destroyed property belonging to his girlfriend, according to the decision issued by the Magistrate regarding his ex-wife's claim."

    prompt_predicted_by_llm = "What were the key arguments presented by the defendant in response to his ex-wife's claims?"

    prompt_translated_to_primary_language = "Yiziphi izimpikiswano eziyinhloko ezethulwe ngumsolwa ephendula izimangalo zowayengumkakhe?"

    instruction_tuning_pair = InstructionTuningPair(
        language, prompt_translated_to_primary_language, primary_language
    )
    print(instruction_tuning_pair)
