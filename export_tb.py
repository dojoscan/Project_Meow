import tensorflow as tf
path_to_events_file = '/Users/Donal/Desktop/Thesis/Code/Project_Meow/meowNET/output/GD_0_1/events.out.tfevents.1487060303.DONAL'
for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if v.tag == 'Accuracy':
            print(v.simple_value)