# rusty-herbarium

Herbarium 2020 using Rust.

The following stats were run on a Lenovo P53s with 46GB RAM.  I used the following Rust-based ML libraries:

* [rusty-machine](https://github.com/AtheMathmo/rusty-machine)
* [rustlearn](https://github.com/maciejkula/rustlearn)
* [juice](https://github.com/spearow/juice)

```
$ ./target/release/serialize_train_and_label_data -b ~/Downloads/herbarium/nybg2020 -o ~/ -w 70 -h 70 -c 40 -l info
2020-04-24 10:46:05,766 INFO  [serialize_train_and_label_data] writing: herbarium-training-labels-70x70.ser.gz
2020-04-24 10:46:05,766 INFO  [serialize_train_and_label_data] writing: herbarium-training-data-70x70.ser.gz
2020-04-24 10:46:07,954 INFO  [serialize_train_and_label_data] writing: herbarium-validation-labels-70x70.ser.gz
2020-04-24 10:46:07,954 INFO  [serialize_train_and_label_data] writing: herbarium-validation-data-70x70.ser.gz
2020-04-24 10:46:07,997 INFO  [serialize_train_and_label_data] Duration: 34s 939ms 225us 52ns
$ ./target/release/juice_mlp_nn -s ~/ -w 70 -h 70 -b 2 -l info
2020-04-24 10:46:47,888 INFO  [juice::layers::container::sequential] Input 0 -> data
2020-04-24 10:46:47,888 INFO  [juice::layers::container::sequential] Creating Layer reshape
2020-04-24 10:46:47,888 INFO  [juice::layer] Input data            -> Layer         reshape
2020-04-24 10:46:47,888 INFO  [juice::layer] Layer reshape         -> Output            data (in-place)
2020-04-24 10:46:47,888 INFO  [juice::layers::container::sequential] Creating Layer linear1
2020-04-24 10:46:47,888 INFO  [juice::layer] Input data            -> Layer         linear1
2020-04-24 10:46:47,888 INFO  [juice::layer] Layer linear1         -> Output    SEQUENTIAL_1
2020-04-24 10:46:47,888 INFO  [juice::layer] Output 0 = SEQUENTIAL_1
2020-04-24 10:46:47,888 INFO  [juice::layer] Layer linear1 - appending weight
2020-04-24 10:46:48,283 INFO  [juice::layers::container::sequential] Creating Layer sigmoid
2020-04-24 10:46:48,283 INFO  [juice::layer] Input SEQUENTIAL_1    -> Layer         sigmoid
2020-04-24 10:46:48,283 INFO  [juice::layer] Layer sigmoid         -> Output    SEQUENTIAL_1 (in-place)
2020-04-24 10:46:48,283 INFO  [juice::layers::container::sequential] Creating Layer linear2
2020-04-24 10:46:48,283 INFO  [juice::layer] Input SEQUENTIAL_1    -> Layer         linear2
2020-04-24 10:46:48,283 INFO  [juice::layer] Layer linear2         -> Output    SEQUENTIAL_3
2020-04-24 10:46:48,283 INFO  [juice::layer] Output 0 = SEQUENTIAL_3
2020-04-24 10:46:48,283 INFO  [juice::layer] Layer linear2 - appending weight
2020-04-24 10:46:48,676 INFO  [juice::layers::container::sequential] Creating Layer linear3
2020-04-24 10:46:48,676 INFO  [juice::layer] Input SEQUENTIAL_3    -> Layer         linear3
2020-04-24 10:46:48,676 INFO  [juice::layer] Layer linear3         -> Output    SEQUENTIAL_4
2020-04-24 10:46:48,676 INFO  [juice::layer] Output 0 = SEQUENTIAL_4
2020-04-24 10:46:48,676 INFO  [juice::layer] Layer linear3 - appending weight
2020-04-24 10:46:48,774 INFO  [juice::layers::container::sequential] Creating Layer linear4
2020-04-24 10:46:48,774 INFO  [juice::layer] Input SEQUENTIAL_4    -> Layer         linear4
2020-04-24 10:46:48,774 INFO  [juice::layer] Layer linear4         -> Output    SEQUENTIAL_5
2020-04-24 10:46:48,774 INFO  [juice::layer] Output 0 = SEQUENTIAL_5
2020-04-24 10:46:48,774 INFO  [juice::layer] Layer linear4 - appending weight
2020-04-24 10:46:48,775 INFO  [juice::layers::container::sequential] Creating Layer log_softmax
2020-04-24 10:46:48,775 INFO  [juice::layer] Input SEQUENTIAL_5    -> Layer     log_softmax
2020-04-24 10:46:48,775 INFO  [juice::layer] Layer log_softmax     -> Output SEQUENTIAL_OUTPUT_6
2020-04-24 10:46:48,775 INFO  [juice::layer] Output 0 = SEQUENTIAL_OUTPUT_6
2020-04-24 10:46:48,775 INFO  [juice::layer] log_softmax needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layer] linear4 needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layer] linear3 needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layer] linear2 needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layer] sigmoid needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layer] linear1 needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layer] reshape needs backward computation: true
2020-04-24 10:46:48,775 INFO  [juice::layers::container::sequential] Sequential container initialization done.
2020-04-24 10:46:48,857 INFO  [juice::layers::container::sequential] Input 0 -> network_out
2020-04-24 10:46:48,857 INFO  [juice::layers::container::sequential] Input 1 -> label
2020-04-24 10:46:48,857 INFO  [juice::layers::container::sequential] Creating Layer nll
2020-04-24 10:46:48,857 INFO  [juice::layer] Input network_out     -> Layer             nll
2020-04-24 10:46:48,857 INFO  [juice::layer] Input label           -> Layer             nll
2020-04-24 10:46:48,857 INFO  [juice::layer] Layer nll             -> Output SEQUENTIAL_OUTPUT_0
2020-04-24 10:46:48,857 INFO  [juice::layer] Output 0 = SEQUENTIAL_OUTPUT_0
2020-04-24 10:46:48,857 INFO  [juice::layer] nll needs backward computation: true
2020-04-24 10:46:48,857 INFO  [juice::layers::container::sequential] Sequential container initialization done.
Accuracy 0/2 = 0.00%
Accuracy 2/4 = 50.00%
Accuracy 4/6 = 66.67%
Accuracy 6/8 = 75.00%
...snip...
Accuracy 932/1000 = 93.20%
Accuracy 932/1000 = 93.20%
Accuracy 932/1000 = 93.20%
Accuracy 932/1000 = 93.20%
2020-04-24 10:48:17,604 INFO  [juice_mlp_nn] Duration: 1m 30s 146ms 379us 796ns
```
