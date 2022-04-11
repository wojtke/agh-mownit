import annealing as an
import zad2_cost_functions as cf
from PIL import Image
import numpy as np

save_dir = 'zad2_imgs/'

cost_functions = [cf.cost1, cf.cost2, cf.cost3, cf.cost4, cf.cost5, cf.cost6]
densities = [0.2, 0.5, 0.7]
sizes = [128, 256]
T_change_functions = [(lambda T:T*0.992),(lambda T:T*0.999)]

def save_img(name, A):
	im = Image.fromarray(A)
	im.save(save_dir+name+'.jpeg')

def save_chart(name, cost_hist, T_hist):
	fig = an.plot_annealing(cost_hist, T_hist)
	fig.write_image(save_dir+name+'.svg')

def gen_random_map(n, density):
	return np.random.uniform(0, 1, size=(n, n))> (1-density)

def swap(A, T):
    A = np.copy(A)
    n_pts = int(len(A)//3 + 2*T)
    pts = np.random.randint(len(A), size=(2,n_pts))
    x, y = np.unique(pts, axis=1)
    per = np.random.permutation(len(x))
    A[x,y] = A[x[per],y[per]]
    return A

if __name__ == '__main__':
	for c, cost_fun in enumerate(cost_functions):
		for d in densities:
			for s in sizes:
				for t, T_change_fun in enumerate(T_change_functions):

					name = f"cost{c+1}-dens{d}-size{s}-cooldown{t+1}"
					print("\n--------------------")
					print(name)

					A = gen_random_map(s, d)

					x, cost_hist, T_hist = an.anneal(
					    x_0 = A,
					    cost_fun = cost_fun,
					    update_fun = swap,
					    T_change_fun = T_change_fun,
					    iters=(2500+2*s)*(t+2),
					    max_retries=50,
					    verbose=True
					)

					print("Saving...")
					save_img(name, x)
					save_chart(name, cost_hist, T_hist)
					print("...done\n")




