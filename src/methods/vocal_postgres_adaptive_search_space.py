from methods.vocal_postgres import VOCALPostgres

class VOCALPostgresAdaptiveSearchSpace(VOCALPostgres):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        while self.max_duration > 1:
            if self.budget - self.n_init_pos - self.n_init_neg < self.max_npred + (self.max_duration - 1) * self.max_depth:
                self.max_duration -= 1
            else:
                break
        samples_per_iter = [0] * (self.max_npred + (self.max_duration - 1) * self.max_depth)
        for i in range((self.budget - self.n_init_pos - self.n_init_neg)):
            # samples_per_iter[len(samples_per_iter) - 1 - i % len(samples_per_iter)] += 1 # Lazy
            # samples_per_iter[i % len(samples_per_iter)] += 1 # Eager
            samples_per_iter[len(samples_per_iter)//2+((i% len(samples_per_iter)+1)//2)*(-1)**(i% len(samples_per_iter))] += 1 # Iterate from the middle
        self.samples_per_iter = samples_per_iter
        print(samples_per_iter)