# CNN

```rust
#[tokio::main]
async fn main() {
    delay::new(2000).await;
    println!("hello, {}", "rust");
}
```

------

## 全连接

### 2.1 前馈传播

在下面的推导中，我们采用平方误差代价函数。我们讨论的是多类问题，共c类，共N个训练样本。

$$E^N = \frac{1}{2} \sum_{n=1}^N \sum_{k=1}^c(t_k^n-y_k^n)^2$$



这里$t_k^n$表示第n个样本对应的标签的第k维。$y_k^n$表示第n个样本对应的网络输出的第k个输出.对于多类问题，输出一般组织为“one-of-c”的形式，也就是只有该输入对应的类的输出节点输出为正，其他类的位或者节点为0或者负数，这个取决于你输出层的激活函数。sigmoid就是0，tanh就是-1.



因为在全部训练集上的误差只是每个训练样本的误差的总和，所以这里我们先考虑对于一个样本的BP。对于第n个样本的误差，表示为：

$$E^n = \frac{1}{2}\sum_{n=1}^c(t^n-y^n)^2$$

传统的全连接神经网络中，我们需要根据BP规则计算代价函数E关于网络每一个权值的偏导数。我们用$\ell$来表示当前层，那么当前层的输出可以表示为：

$$ x^\ell=f(u^\ell), with\space u^\ell=W^\ell x^{\ell-1}+b^\ell $$

输出激活函数$f(·)$可以有很多种，一般是sigmoid函数或者双曲线正切函数。sigmoid将输出压缩到[0, 1]，所以最后的输出平均值一般趋于0 。所以如果将我们的训练数据归一化为零均值和方差为1，可以在梯度下降的过程中增加收敛性。对于归一化的数据集来说，双曲线正切函数也是不错的选择。

### 2.2 反向传播

反向传播回来的误差可以看做是每个神经元的基的灵敏度sensitivities（灵敏度的意思就是我们的基b变化多少，误差会变化多少，也就是误差对基的变化率，也就是导数了），定义如下：（第二个等号是根据求导的链式法则得到的）。

$$\frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}\frac{\partial u}{\partial b}=\delta$$

因为$\frac{\partial u}{\partial b} = 1$. 也就是说bias基的灵敏度$\frac{\partial E}{\partial b}=\delta$和误差E对一个节点全部输入u的导数$\frac{\partial E}{\partial u}$是相等的。**正是这个导数使得高层到底层的反向传播**， 迭代表达式

$$\delta^\ell=(W^{\ell+1})^T\delta^{\ell+1}\circ f'(u^\ell)$$

 这里的$\circ$表示每个元素相乘。

 推导过程如下 $nl$层为输出层

1.  Perform a feedforward pass, computing the activations for layers $L_2$, $L_3$, and so on up to the output layer $L_{nl}$.


1.  For each output unit $i$ in layer $nl$ (the output layer), set:
    
    $$\delta^{(n_l)}_i= \frac{\partial}{\partial z^{(n_l)}_i} \;\;\frac{1}{2} \left\|y - h_{W,b}(x)\right\|^ 2 = - (y_i - a^{(n_l)}_i) \cdot f'(z^{(n_l)}_i)$$
    
2.  For $l=nl−1,nl−2,nl−3,…,2$, for each node $i$ in layer $l$, set:
    
    $$\delta_i^{(l)}=\bigg(\sum_{j=1}^{sl+1}W_{ji}^{(l)}\delta_j^{(l+1)}\bigg)f'(z_i^{(l)})$$

证明上式：

![证明](http://img.blog.csdn.net/20150620215526135?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTHU1OTcyMDM5MzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



对于误差函数$E^n = \frac{1}{2}\sum_{n=1}^c(t^n-y^n)^2$, 输出层的神经元的灵敏度是不一样的：

$$\delta^L=f'(u^L)\circ(y^n-t^n)$$

最后，对每个神经元运用$\delta$规则进行权值更新。具体来说就是，对一个给定的神经元，得到它的输入，然后用这个神经元的$\delta$来进行缩放。用向量的形式表述就是，对于第$\ell$层，误差对于该层每一个权值（组合为矩阵）的导数是该层的输入（等于上一层的输出）与该层的灵敏度（该层每个神经元的δ组合成一个向量的形式）的叉乘。然后得到的偏导数乘以一个负学习率就是该层的神经元的权值的更新了：

$$\frac{\partial E}{\partial W^\ell}=x^{\ell-1}(\delta^\ell)^T$$

$$ \Delta W^\ell=-\eta \frac{\partial E}{\partial W^\ell}$$



证明

![](http://img.blog.csdn.net/20150620215644802?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTHU1OTcyMDM5MzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

对于bias基的更新表达式差不多。实际上，对于每一个权值$W_{ij}$都有一个特定的学习率$\eta_{ij}$。



### 3. Convolutional Neural Networks

我们现在关注网络中卷积层的BP更新。在一个卷积层，上一层的特征maps被一个可学习的卷积核进行卷积，然后通过一个激活函数，就可以得到输出特征map。每一个输出map可能是组合卷积多个输入maps的值：

$$x_j^\ell=f\bigg(\sum_{i\in M_j} x_i^{\ell-1}*k_{ij}^\ell+b_j^\ell \bigg)$$



where, 这里$M_j$表示选择的输入maps的集合，那么到底选择哪些输入maps呢？有选择一对的或者三个的。但下面我们会讨论如何去自动选择需要组合的特征maps。每一个输出map会给一个额外的偏置b，但是对于一个特定的输出map，卷积每个输入maps的卷积核是不一样的。也就是说，如果输出特征map j和输出特征map k都是从输入map i中卷积求和得到，那么对应的卷积核是不一样的。

#### 3.1.1 Computing the Gradients

我们假定每个卷积层$\ell$都会接一个下采样层$\ell+1$。对于BP来说，根据上文我们知道，要想求得$\ell$层的每个神经元对应的权值的权值更新，就需要先求层$\ell$的每一个神经节点的灵敏度$\delta$（也就是上面的权值更新的公式）。为了求这个灵敏度我们就需要先对下一层的节点（连接到当前层$\ell$的感兴趣节点的第$\ell+1$层的节点）的灵敏度求和（得到$\delta^{\ell+1}$），然后乘以这些连接对应的权值（连接第$\ell$层感兴趣节点和第$\ell+1$层节点的权值）W。再乘以当前层$\ell$的该神经元节点的输入u的激活函数f的导数值，这样就可以得到当前层l每个神经节点对应的灵敏度$\delta^\ell$了。(此处的公式如下：）

$$\delta^\ell=(W^{\ell+1})^T\delta^{\ell+1}\circ f'(u^\ell)$$

然而因为sample的存在，采样层的一个像素（神经元节点）对应的灵敏度δ对应于卷积层（上一层）的输出map的一块像素（采样窗口大小）。

![sample](http://www.36dsj.com/wp-content/uploads/2015/03/6.gif)

因此，层$\ell$中的一个map的每个节点只与$\ell+1$层中相应map的一个节点连接。

为了有效计算层l的灵敏度，我们需要upsample这个downsample层对应的灵敏度map（特征map中每个像素对应一个灵敏度，所以也组成一个map），这样才使得这个灵敏度map大小与卷积层的map大小一致，然后再将层$\ell$的map的激活值的偏导数与从第$\ell+1$层的上采样得到的灵敏度map逐元素相乘,就是公式$\delta^\ell=(W^{\ell+1})^T\delta^{\ell+1}\circ f'(u^\ell)$。 在downsample层map的权值都取一个相同值$\beta$，而且是一个常数。所以我们只需要将上一个步骤得到的结果乘以一个$\beta$就可以完成第$\ell$层灵敏度$\delta^\ell$的计算。我们可以对每个卷积层的map $j$重复这个计算过程,需要匹配相应的子采样层的map:

$$\delta_j^\ell=\beta_j^{\ell+1}\bigg(f'(u_j^\ell)\circ up(\delta_j^{\ell+1})\bigg)$$

其中 up(·) 是upsample 操作。如果下采样的采样因子是n的话，它简单的将每个像素水平和垂直方向上拷贝n次。这样就可以恢复原来的大小了。实际上，这个函数可以用Kronecker乘积来实现, 也就是说：

$$up(x)\equiv x\otimes 1_{n\times n}$$



![克罗克内积](http://c.hiphotos.baidu.com/baike/c0%3Dbaike116%2C5%2C5%2C116%2C38/sign=7dde9e4076c6a7efad2ba0749c93c434/d009b3de9c82d1585670622e800a19d8bd3e42ba.jpg)

``` cpp
// 克罗克内积,对矩阵进行扩展
void kronecker(
        const vector2d& matrix,
        const Size& scale,
        vector2d& result
    ) {
    const int m = matrix.size();
    int n = matrix[0].size();

    vector1d temp1d;
    temp1d.resize(n * scale.y);
    result.resize(m * scale.x, temp1d);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int ki = i * scale.x; ki < (i + 1) * scale.x; ki++) {
                for (int kj = j * scale.y; kj < (j + 1) * scale.y; kj++) {
                    result[ki][kj] = matrix[i][j];
                }
            }
        }
    }
}
```

 好，到这里，对于一个给定的map，我们就可以计算得到其灵敏度map了。我们就可以直接通过简单的对$\delta_j^\ell$中的所有元素进行求和快速的计算bias基的梯度了：

$$\frac{\partial E}{\partial b_j}=\sum_{u,v} (\delta_j^\ell)_{uv}$$



``` cpp
void CNN::updateBias(Layer& layer) {
	vector4d& errors = layer.getErrors();
    int lengthI = errors.size();
    int lengthJ = errors[0].size(); // outMapNum
    int lengthK = errors[0][0].size();
    int lengthL = errors[0][0][0].size();

    for (int j = 0; j < lengthJ; j++) {
        double errorSum = 0;
        for (int i = 0; i < lengthI; ++i) 
            for (int k = 0; k < lengthK; ++k) 
                for (int l = 0; l < lengthL; ++l) 
                    errorSum += errors[i][j][k][l];

        // 更新偏置
        double deltaBias =  errorSum / batchSize;
        double bias = layer.getBias(j) + ALPHA * deltaBias;
        layer.setBias(j, bias);
    }

}
```

最后，计算核权重的梯度，可以用BP算法来计算. $\frac{\partial E}{\partial W^\ell}=x^{\ell-1}(\delta^\ell)^T$. 另外，很多连接的权值是共享的，因此，对于一个给定的权值，我们需要对所有与该权值有联系（权值共享的连接）的连接对该点求梯度，然后对这些梯度进行求和，就像上面对bias基的梯度计算一样：

$$\frac{\partial E}{\partial k_{ij}^\ell}=\sum_{u,v}(\delta_j^\ell)_{uv}(p_i^{\ell-1})_{uv}$$

这里，$(p_i^{\ell-1})_{uv}$是$x_i^{\ell-1}$中的在卷积的时候与$k_{ij}^\ell$逐元素相乘的patch，输出卷积map的(u, v)位置的值是由上一层的(u, v)位置的patch与卷积核$k_{ij}$逐元素相乘的结果。 对于上面的公式，可以用Matlab的卷积函数来实现： 

$$\frac{\partial E}{\partial k_{ij}^\ell}=rot180(conv2(x_i^{\ell-1},rot180(\delta_j^\ell),'valid'))$$



``` cpp
void CNN::updateKernels(Layer& layer, Layer& lastLayer) {
    int mapNum = layer.getOutMapNum();
    const int lastMapNum = lastLayer.getOutMapNum();

    for (int j = 0; j < mapNum; j++) {
        for (int i = 0; i < lastMapNum; i++) {
            // 对batch的每个记录delta求和
            vector2d deltaKernel;
            for (int r = 0; r < batchSize; r++) {
                vector2d& error = layer.getError(r, j);
                if (deltaKernel.size() == 0) {
                    convnValid(lastLayer.getMap(r, i), error, deltaKernel);
                } else { // 累积求和
                    vector2d temp;
                    // $$\frac{\partial E}{\partial k_{ij}^\ell}$$
                    convnValid(lastLayer.getMap(r, i), error, temp);

                    ProcessTwo plusFunc = &plus;
                    matrixOp(temp, deltaKernel, deltaKernel, NULL, NULL, plusFunc);
                }
            }

            // 除以batchSize
            matrixOp(deltaKernel, deltaKernel, divideBatchSize);
            // 更新卷积核
            vector2d& kernel = layer.getKernel(i, j);

            ProcessTwo plusFunc = &plus;
            vector2d temp2d;
            vector1d temp1d;
            temp1d.resize(deltaKernel[0].size());
            temp2d.resize(deltaKernel.size(), temp1d);

            matrixOp(kernel, deltaKernel, temp2d, multiplyLambda, multiplyAlpha, plusFunc);

            layer.setKernel(i, j, temp2d);
        }
    }
}
```

### 3.2 Sub-sampling Layers

对于子采样层来说，有N个输入maps，就有N个输出maps，尽管每个输出map可能都变小了。通常：

$$x_j^\ell=f\bigg(\beta_j^\ell down(x_j^{\ell-1})+b_j^\ell \bigg)$$

其中 down(·) 表示一个 sub-sampling 函数。典型的操作一般是对输入图像的不同$n \times n$的块的所有像素进行求和。这样输出图像在两个维度上都缩小了n倍。每个输出map都对应一个属于自己的乘性偏置$\beta$和一个加性偏置b。



##### 3.2.1 Computing the Gradients

困难点在于计算灵敏度map。一旦我们得出，那我们唯一需要更新的偏置参数$\beta$和b就可以轻而易举了。 （$\frac{\partial E}{\partial b_j}=\sum_{u,v} (\delta_j^\ell)_{uv}$）。如果下一个卷积层与这个子采样层是全连接的，那么就可以通过BP来计算子采样层的灵敏度maps。

当我们计算卷积核的梯度，所以我们必须找到输入map中哪个patch对应输出map的哪个像素。这里，就必须找到当前层的灵敏度map中哪个patch对应与下一层的灵敏度map的给定像素，这样才可以使用$\delta$递推。

$$\delta_j^\ell=f'(u_j^\ell)\circ \text{conv2}(\delta_j^{\ell+1}, \text{rot180}(k_j^{\ell+1}),'\text{full}')$$

计算之前之前，我们需要先将核旋转一下，让卷积函数可以实施互相关计算。另外，我们需要对卷积边界进行处理，在Matlab里面，就比较容易处理。Matlab中全卷积会对缺少的输入像素补0 。

​	到这里，我们就可以对b和$\beta$计算梯度了。首先，加性基b的计算和上面卷积层的一样，对灵敏度map中所有元素加起来就可以了

$$\frac{\partial E}{\partial b_j}=\sum_{u,v}(\delta_j^\ell)_{uv}$$

 而对于乘性偏置β，因为涉及到了在前向传播过程中下采样map的计算，所以我们最好在前向的过程中保存好这些maps，这样在反向的计算中就不用重新计算了。我们定义：

$$\text{d}_j^\ell=\text{down}(\text{x}_j^{\ell-1})$$

那么关于$\beta$的梯度为

$$\frac{\partial E}{\partial \beta_j}=\sum_{u,v}(\delta_j^\ell \circ \text{d}_j^\ell)_{uv}$$



``` cpp
void CNN::setSampErrors(Layer& layer, Layer& nextLayer) {
    int mapNum = layer.getOutMapNum();
    const int nextMapNum = nextLayer.getOutMapNum();

    for (int i = 0; i < mapNum; i++) {
        vector2d sum; // 对每一个卷积进行求和
        for (int j = 0; j < nextMapNum; j++) {
            vector2d& nextError = nextLayer.getError(j);
            vector2d& kernel = nextLayer.getKernel(i, j);

            // 对卷积核进行180度旋转，然后进行full模式下得卷积
            vector2d kernelRot180;
            rot180(kernel, kernelRot180);

            if (sum.size() == 0) {
                convnFull(nextError, kernelRot180, sum);
            } else {
                vector2d convnResult;
                convnFull(nextError, kernelRot180, convnResult);
                ProcessTwo plusFunc = &plus;
                matrixOp(convnResult, sum, sum, NULL, NULL, plusFunc);
            }
            layer.setError(i, sum);
        }
    }
}


void convnFull(vector2d& matrix, const vector2d& kernel, vector2d& result) {
    int m = matrix.size();
    int n = matrix[0].size();
    const int km = kernel.size();
    const int kn = kernel[0].size();
    // 扩展矩阵
    vector1d temp1d;
    vector2d extendMatrix;
    temp1d.resize(n + 2 * (kn - 1));
    extendMatrix.resize(m + 2 * (km - 1), temp1d);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
        }
    }

    convnValid(extendMatrix, kernel, result);
}
```





### 3.3  Learning Combinations of Feature Maps

大部分时候，通过卷积多个输入maps，然后再对这些卷积值求和得到一个输出map，这样的效果往往是比较好的。在一些文献中，一般是人工选择哪些输入maps去组合得到一个输出map。但我们这里尝试去让CNN在训练的过程中学习这些组合，也就是让网络自己学习挑选哪些输入maps来计算得到输出map才是最好的。我们用$\alpha_{ij}$表示在得到第j个输出map的其中第i个输入map的权值。这样，第j个输出map可以表示为：

$$\text{x}_j^\ell=f\bigg(\sum_{i=1}^{N_{in}} \alpha_{ij}(\text{x}_i^{\ell-1}*\text{k}_i^\ell)+\text{b}_j^\ell \bigg)$$

其中 $\sum_i\alpha_{ij}=1$ 并且 $0 \leq \alpha_{ij} \leq 1$ 。 这些对变量$\alpha_{ij}$的约束可以通过将变量$\alpha_{ij}$表示为一个组无约束的隐含权值$c_{ij}$的softmax函数来加强。（因为softmax的因变量是自变量的指数函数，他们的变化率会不同）。

$$\alpha_{ij}=\frac{\text{exp}(c_{ij})}{\sum_k\text{exp}(c_{kj})}$$

 因为对于一个固定的j来说，每组权值$c_{ij}$都是和其他组的权值独立的，所以为了方面描述，我们把下标j去掉，只考虑一个map的更新，其他map的更新是一样的过程，只是map的索引j不同而已。

Softmax函数的导数表示为：

$$\frac{\partial \alpha_k}{\partial c_i}=\delta_{ki}\alpha_i-\alpha_i\alpha_k$$

这儿 $\delta$ 是 Kronecker delta。那么$\ell$层的

$$\frac{\partial E}{\partial \alpha_i}=\frac{\partial E}{\partial u^\ell}\frac{\partial u^\ell}{\partial \alpha_i}=\sum_{u,v}\big(\delta^\ell \circ (x_i^{\ell-1}*k_i^\ell)\big) $$

最后就可以通过链式规则去求得代价函数关于权值ci的偏导数了：

$$\frac{\partial E}{\partial c_i}=\sum_k \frac{\partial E}{\partial \alpha_k}\frac{\partial \alpha_k}{\partial c_i} $$

$$= \alpha_i \bigg(\frac{\partial E}{\partial \alpha_i}-\sum_k\frac{\partial E}{\partial \alpha_k} \alpha_k \bigg )$$



#### 3.3.1  Enforcing Sparse Combinations

为了限制 $\alpha_i$是稀疏的，也就是限制一个输出map只与某些而不是全部的输入maps相连。我们在整体代价函数里增加稀疏约束项$\Omega(\alpha)$。对于单个样本，重写代价函数为

$$\tilde{E}^n=E^n+\lambda\sum_{i,j}|(\alpha)_{i,j}|$$

然后寻找这个规则化约束项对权值ci求导的贡献。规则化项$\Omega(\alpha)$对$\alpha_i$求导是：

$$\frac{\partial \Omega}{\partial \alpha_i} =\lambda \text{sign}(\alpha_i)$$

通过链式法则，对ci的求导是：

$$\frac{\partial \Omega}{\partial c_i} =\sum_k \frac{\partial \Omega}{\partial \alpha_k}\frac{\partial \alpha_k}{\partial c_i} = \lambda \bigg(|\alpha_i|-\alpha_i\sum_k |\alpha_k|\bigg)$$

所以，权值ci最后的梯度是：

$$\frac{\partial \tilde{E}^n}{\partial c_i}=\frac{\partial E^n}{\partial c_i} + \frac{\partial \Omega}{\partial c_i}$$

