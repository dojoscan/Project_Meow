import tensorflow as tf
path_to_events_file = '/Users/Donal/Dropbox/CIFAR10/optimizer accuracy/Adam_0_0001/events.out.tfevents.1487150826.PCLUCIA'
for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if v.tag == 'Accuracy':
            print(v.simple_value)