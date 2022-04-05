import math
import random


def base_attribute(obj_dict, num_obs, predicate, a=1, b=0.99):
    """
    Simulate a noisy model, whose accuracy is initially 0.5 and increases up to ``b'' as the number of observations increases.
    Hyperparameters:
        a: set the learning rate of the model. With a greater ``a'', the model accuracy increases more after every user intervention. Default: 1
        b: set the maximum accuracy the model can learn.
    accuracy = b - (b - 0.5) * exp(-0.17 * a * num_obs)
    Parameters
    ----------
    obj_dict: dictionary
    {
        "object_id": 0,
        "color": "blue",
        "material": "rubber",
        "shape": "sphere"
        }
    a: float
    b: float
    num_obs: int
        number of user-labeled samples

    Returns
    -------
    pred: boolean
        True or False, indicating whether the object is of color red.
    """
    rnd = random.uniform(0, 1)
    accuracy = b - (b - 0.5) * math.exp(-0.17 * a * num_obs)
    if predicate(obj_dict):
        return rnd < accuracy
    else:
        return rnd > accuracy

def color_red(obj_dict, num_obs, a=1, b=0.99):
    def predicate(obj_dict):
        return obj_dict["color"] == "red"
    return base_attribute(obj_dict, num_obs, predicate, a, b)

def color_gray(obj_dict, num_obs, a=1, b=0.99):
    def predicate(obj_dict):
        return obj_dict["color"] == "gray"
    return base_attribute(obj_dict, num_obs, predicate, a, b)

def material_metal(obj_dict, num_obs, a=1, b=0.99):
    def predicate(obj_dict):
        return obj_dict["material"] == "metal"
    return base_attribute(obj_dict, num_obs, predicate, a, b)

def collision(sub_id, obj_id, fid, collision_list, num_obs, a=1, b=0.99):
    rnd = random.uniform(0, 1)
    accuracy = b - (b - 0.5) * math.exp(-0.17 * a * num_obs)
    accuracy = 0.99
    for collision_dict in collision_list:
        if set([sub_id, obj_id]) == set(collision_dict["object_ids"]) and collision_dict["frame_id"] == fid:
            return rnd < accuracy
    return rnd > accuracy
