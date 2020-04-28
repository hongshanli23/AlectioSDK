def compute_class_weight(counter, num_classes, min_count):
    '''Calculate class weight based on number of counts per class
    counter: (Counter) key is class index, value is the number of occurence 
        of that class
    num_classes: total number of classes
    if one class does not show up its weight is automatically 0
    '''
    class_weight = [0 for _ in range(num_classes)]

    prod = 1
    for ix in counter:
        prod*=counter[ix]
    
    for ix in counter:
        class_weight[ix] = float(prod) / float(counter[ix])

    return class_weight

