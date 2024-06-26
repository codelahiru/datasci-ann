import random2

marks = [20,35,78,91,10,56,79,16]
grade = [0,0,1,1,0,1,1,0] # supervised training. 0 > Fail and 1 > Pass. Passmark is 50.

#w1 = random2.random()
w1 = 10 # -------------------- increased weight and random disabled
n = 0.001 # ------------------------ reduced the value of n.  n is a small constant value called the 'Eta'   η ( ~ 0.01)
epoch = 0
while epoch < 1000: # ----------------- increased the epoch number

    i=0
    while i < 8:
      tr_in = marks[i]
      net = tr_in*w1
      if net>0.5:
        neuron_out = 1
      else:
        neuron_out = 0

      tr_out = grade[i]
      print(tr_out, neuron_out) # tr_out is the expected o/p

      # to adjust the weights
      neuron_err = tr_out - neuron_out
      del_w = neuron_err * tr_in * n * w1
      w1 = w1 + del_w

      i = i + 1

    print("Epoch: ", epoch, "Weight: ", w1, "Net: ", net, "Delta w: ", del_w)
    epoch = epoch + 1


