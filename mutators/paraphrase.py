class ParaphraseMutator:
    def mutate(self, prompt):
        variants = []
        if "how" in prompt.lower():
            variants.append(
                {"text": prompt.replace("how", "what is the best way to"), "mutation_type": "paraphrase"}
            )
        variants.append(prompt.upper())
        variants.append(prompt.lower())
        return variants
