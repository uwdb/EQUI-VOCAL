from sklearn.metrics import f1_score
import resource
import random

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '''%s: mem=%s MB
           '''%(point, usage/1024.0 )

random.seed(10)

class BaseMethod:
    def compute_query_score(self, current_query):
        y_pred = []
        # for i, (input, label) in enumerate(zip(self.inputs, self.labels)):
        for i in self.labeled_index:
            input = self.inputs[i]
            label = self.labels[i]
            if self.lock:
                self.lock.acquire()
            memoize = self.memoize_all_inputs[i]
            if self.lock:
                self.lock.release()
            result, new_memoize  = current_query.execute(input, label, memoize, {})
            y_pred.append(int(result[0, len(input[0])] > 0))
            if self.lock:
                self.lock.acquire()
            self.memoize_all_inputs[i].update(new_memoize)
            if self.lock:
                self.lock.release()
        # print(self.labels[self.labeled_index], y_pred)
        print("cache", len(self.memoize_all_inputs[0]))
        print(using("profile"))
        score = f1_score(list(self.labels[self.labeled_index]), y_pred)
        return score