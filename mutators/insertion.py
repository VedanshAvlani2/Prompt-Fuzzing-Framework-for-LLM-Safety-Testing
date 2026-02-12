class InsertionMutator:
    def mutate(self, prompt):
        ins = ["as an expert,", "for educational purposes,", "hypothetically,"]
        return [
            {"text": f"{i} {prompt}", "mutation_type": "insertion"}
            for i in ins
        ]
