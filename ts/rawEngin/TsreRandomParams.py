# -*- coding: utf-8 -*-
import numpy as np

class TsreRandomParams():
    def __init__(self, aug_params):
        self.params = aug_params
        self.rng = np.random.default_rng()

    def get_normal(self, min_val, max_val, mean, sigma, scale):
        return np.clip(self.rng.normal(mean, sigma) * scale, min_val, max_val)

    def get_beta_normal(self,  min_val, max_val, scale):
        return np.clip((self.rng.beta(2.0, 2.0) - 0.5) * 2.0 * scale, min_val, max_val)

    def get_temperature_normal_1(self, min_val, max_val, mean_val):
        if np.random.random() < 0.5:
            val = mean_val - (np.clip(np.abs(self.rng.normal(0, 0.3)), 0, 1.0) * (mean_val - min_val) / 1.0)
        else:
            val = mean_val + (np.clip(np.abs(self.rng.normal(0, 0.15)), 0, 1.0) * (max_val - mean_val) / 1.0)
        return val

    def get_temperature_normal_2(self, min_val, max_val, mean_val):
        if np.random.random() < 0.25:
            val = mean_val - (np.clip(np.abs(self.rng.beta(2.0, 2.0) - 0.5) * 2.0, 0, 1.0) * (mean_val - min_val))
        else:
            val = mean_val + (np.clip(np.abs(self.rng.beta(2.0, 2.0) - 0.5) * 2.0, 0, 1.0) * (max_val - mean_val))
        return val

    def __call__(self):
        if self.params is None and len(self.params) == 0:
            return []
        else:
            dict_param = dict()
            for key, min_val, max_val, detail in self.params:
                if (min_val is None) or (max_val is None):
                    # print("Not Support aug params ", key, min_val, max_val)
                    continue
                else:
                    if isinstance(detail, list):
                        threshold, proc_type, params = detail
                        if proc_type == 'normal_1':
                            if np.random.random() < threshold:
                                dict_param[key] = int(self.get_normal(min_val, max_val, 0, 0.3, params[0]))
                        if proc_type == 'bete_normal_1':
                            if np.random.random() < threshold:
                                dict_param[key] = int(self.get_beta_normal(min_val, max_val, params[0]))
                        elif proc_type == 'temperature_normal_1':
                            if np.random.random() < threshold:
                                dict_param[key] = int(self.get_temperature_normal_1(min_val, max_val, params[0]))
                        elif proc_type == 'temperature_normal_2':
                            if np.random.random() < threshold:
                                dict_param[key] = int(self.get_temperature_normal_2(min_val, max_val, params[0]))
                        else:
                            print("Not Support detail", detail)
                            continue
                    elif isinstance(detail, float):
                        if(np.random.random() < detail):
                            dict_param[key] = np.random.randint(min_val, max_val)
                    else:
                        print("Not Support detail", detail)
                        continue
            return dict_param

if __name__ == '__main__':

    random_params_list = TsreRandomParams([
        ["Temperature", 2000, 50000, [1.0, 'temperature_normal_2', [6500]]],
        ["HSLTunnerSaturationRed", -20, 5, 1.0],
    ])

    # Test
    data = [random_params_list()["HSLTunnerSaturationRed"] for count in range(10000)]

    print("avg = ", np.average(data))
    print("min = ", np.min(data))
    print("max = ", np.max(data))

    import matplotlib.pyplot as plt
    plt.hist(data, bins=100, color='red', histtype='stepfilled', alpha=0.75)
    plt.title('Histogram')
    plt.show()
