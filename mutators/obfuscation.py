class ObfuscationMutator:
    def mutate(self, prompt):
        return [
            {"text": prompt.replace("a", "@"), "mutation_type": "obfuscation"},
            {"text": " ".join(prompt), "mutation_type": "obfuscation"},
        ]
