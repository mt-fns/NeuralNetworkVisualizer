# Neural Network Visualizer
A digit recognition neural network made from scratch (no high-level libraries were used, only NumPy and Pandas) trained on the MNIST dataset. I created this to grasp the mathematics behind a neural network, partially inspired by 3Blue1Brown's video series.

</br>
<img width="930" alt="Screen Shot 2024-04-18 at 10 27 15" src="https://github.com/mt-fns/NeuralNetworkVisualizer/assets/80404890/6b4b8ed3-c001-4f0d-bb2c-aedcf13ff575">

# Setup
Simply install the dependencies for this project using pip and run 
```python visualizer.py``` 
to activate the visualizer. This file extracts weights and biases from ```saved_nn.npz``` which stores parameters from previously trained neural networks using ```train.py```. The neural network implementation itself is contained in ```main.py```

# Resources
Some free resources on neural networks that helped me in completing this project:

1. 3Blue1Brown's neural network video series: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
2. The Coding Train's neural network visualizer implementation: https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh
3. Samson Zhang's brief introduction to neural networks: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=41s



