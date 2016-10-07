import matplotlib.pyplot as plt

f = open('plots/relu.txt','r')
a = []
for line in f:
	line = line.split()
	a.append(float(line[0]))
print a
f.close()

f = open('plots/tanh.txt','r')
b = []
for line in f:
        line = line.split()
        b.append(float(line[0]))
print b

plt.figure()

axes = plt.gca()

plt.plot(a,'-ro')
plt.plot(b,'-go')

plt.title("% Accuracy vs Epoch")
plt.xlabel('EPOCH')
plt.ylabel('%Accuracy')

plt.show()
