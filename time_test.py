from fisher_information import * 
import pandas as pd 


# varying wire time test
'''
max_wires = 10
wire_range = range(1, max_wires+1)
exact = []
approx100 = []
approx200 = []
approx300 = []

for n_wires in wire_range:
	wires = n_wires
	n_params = wires * 4
	params = np.random.uniform(0, 2 * np.pi, n_params)

	# exact calculation
	start = time.time()
	fisher_matrix = compute_fisher_matrix(PQC_function, wires, params)
	exact.append(time.time() - start)

	# 100 shot approximation
	start = time.time()
	fisher_matrix = compute_fisher_matrix_spsa(PQC_function, wires, params, shots=100)
	approx100.append(time.time()-start)

	# 200 shot approximation
	start = time.time()
	fisher_matrix = compute_fisher_matrix_spsa(PQC_function, wires, params, shots=200)
	approx200.append(time.time()-start)

	# 300 shot approximation
	start = time.time()
	fisher_matrix = compute_fisher_matrix_spsa(PQC_function, wires, params, shots=300)
	approx300.append(time.time()-start)

	print(f'{n_wires} wire(s) done!')

# save time test data
df = pd.DataFrame({'exact': exact,
	'approx100': approx100,
	'approx200': approx200,
	'approx300': approx300,
	})
df.to_csv('time_test.csv')

# plot time test data
plt.figure(figsize = (7,5))

plt.title('Running time of exact calculation and approximation')
plt.xlabel('Number of wires')
plt.ylabel('Running time (s)')

plt.plot(wire_range, exact, label='Exact calculation')
plt.plot(wire_range, approx100, label='100 shot approximation')
plt.plot(wire_range, approx200, label='200 shot approximation')
plt.plot(wire_range, approx300, label='300 shot approximation')
plt.legend()

plt.savefig('time_test.jpg')
'''

# varying layer time test

wires = 1

max_layers = 20
layer_range = range(1, max_layers+1)
exact = []
approx100 = []
approx200 = []
approx300 = []

for n_layers in layer_range:
	layers = n_layers
	n_params = 2*wires*(1 + n_layers)
	params = np.random.uniform(0, 2 * np.pi, n_params)

	# exact calculation
	start = time.time()
	fisher_matrix = compute_fisher_matrix(PQC_function, wires, params, layers)
	exact.append(time.time() - start)

	# 100 shot approximation
	start = time.time()
	fisher_matrix = compute_fisher_matrix_spsa(PQC_function, wires, params, 100, layers)
	approx100.append(time.time() - start)

	# 200 shot approximation
	start = time.time()
	fisher_matrix = compute_fisher_matrix_spsa(PQC_function, wires, params, 200, layers)
	approx200.append(time.time() - start)

	# 300 shot approximation
	start = time.time()
	fisher_matrix = compute_fisher_matrix_spsa(PQC_function, wires, params, 300, layers)
	approx300.append(time.time() - start)

	print(f'{layers} layer(s) done!')

# save time test data
df = pd.DataFrame({'exact': exact,
	'approx100': approx100,
	'approx200': approx200,
	'approx300': approx300,
	})
df.to_csv('layer_time_test.csv')

# plot time test data
plt.figure(figsize = (7,5))

plt.title(f'Running time with {wires} wire(s) and varying number of layers')
plt.xlabel('Number of layers')
plt.ylabel('Running time (s)')

plt.plot(layer_range, exact, label='Exact calculation')
plt.plot(layer_range, approx100, label='100 shot approximation')
plt.plot(layer_range, approx200, label='200 shot approximation')
plt.plot(layer_range, approx300, label='300 shot approximation')
plt.legend()

plt.xticks([4,8,12,16,20])

plt.savefig('layer_time_test.jpg')