## 一、前言
---
本篇文章对于什么是SVM做出了生动的阐述，同时也进行了线性SVM的理论推导，以及最后的编程实践，公式较多，还需静下心来一点一点推导。

本文出现的所有代码，均可在我的github上下载，欢迎Follow、Star：[github地址](https://github.com/yaoguangju/machine_learning)

## 二、什么是SVM？
---
SVM的英文全称是Support Vector Machines，我们叫它支持向量机。支持向量机是我们用于分类的一种算法。让我们以一个小故事的形式，开启我们的SVM之旅吧。

在很久以前的情人节，一位大侠要去救他的爱人，但天空中的魔鬼和他玩了一个游戏。

魔鬼在桌子上似乎有规律放了两种颜色的球，说："你用一根棍分开它们？要求：尽量在放更多球之后，仍然适用。"

![](img/1.png)

于是大侠这样放，干的不错？

![](img/2.png)

然后魔鬼，又在桌上放了更多的球，似乎有一个球站错了阵营。显然，大侠需要对棍做出调整。

![](img/3.png)

SVM就是试图把棍放在最佳位置，好让在棍的两边有尽可能大的间隙。这个间隙就是球到棍的距离。

![](img/4.png)

现在好了，即使魔鬼放了更多的球，棍仍然是一个好的分界线。

![](img/5.png)

魔鬼看到大侠已经学会了一个trick(方法、招式)，于是魔鬼给了大侠一个新的挑战。

![](img/6.png)

现在，大侠没有棍可以很好帮他分开两种球了，现在怎么办呢？当然像所有武侠片中一样大侠桌子一拍，球飞到空中。然后，凭借大侠的轻功，大侠抓起一张纸，插到了两种球的中间。

![](img/7.png)

现在，从空中的魔鬼的角度看这些球，这些球看起来像是被一条曲线分开了。

![](img/8.png)

再之后，无聊的大人们，把这些球叫做data，把棍子叫做classifier,找到最大间隙的trick叫做optimization，拍桌子叫做kernelling, 那张纸叫做hyperplane。

更为直观地感受一下吧(需要梯子)：https://www.youtube.com/watch?v=3liCbRZPrZA

概述一下：

当一个分类问题，数据是线性可分的，也就是用一根棍就可以将两种小球分开的时候，我们只要将棍的位置放在让小球距离棍的距离最大化的位置即可，寻找这个最大间隔的过程，就叫做最优化。但是，现实往往是很残酷的，一般的数据是线性不可分的，也就是找不到一个棍将两种小球很好的分类。这个时候，我们就需要像大侠一样，将小球拍起，用一张纸代替小棍将小球进行分类。想要让数据飞起，我们需要的东西就是核函数(kernel)，用于切分小球的纸，就是超平面。

也许这个时候，你还是似懂非懂，没关系。根据刚才的描述，可以看出，问题是从线性可分延伸到线性不可分的。那么，我们就按照这个思路，进行原理性的剖析。

## 三、线性SVM
---
先看下线性可分的二分类问题。

![](img/9.png)

上图中的(a)是已有的数据，红色和蓝色分别代表两个不同的类别。数据显然是线性可分的，但是将两类数据点分开的直线显然不止一条。上图的(b)和(c)分别给出了B、C两种不同的分类方案，其中黑色实线为分界线，术语称为“决策面”。每个决策面对应了一个线性分类器。虽然从分类结果上看，分类器A和分类器B的效果是相同的。但是他们的性能是有差距的，看下图：

![](img/10.png)

在"决策面"不变的情况下，我又添加了一个红点。可以看到，分类器B依然能很好的分类结果，而分类器C则出现了分类错误。显然分类器B的"决策面"放置的位置优于分类器C的"决策面"放置的位置，SVM算法也是这么认为的，它的依据就是分类器B的分类间隔比分类器C的分类间隔大。这里涉及到第一个SVM独有的概念"分类间隔"。在保证决策面方向不变且不会出现错分样本的情况下移动决策面，会在原来的决策面两侧找到两个极限位置（越过该位置就会产生错分现象），如虚线所示。虚线的位置由决策面的方向和距离原决策面最近的几个样本的位置决定。而这两条平行虚线正中间的分界线就是在保持当前决策面方向不变的前提下的最优决策面。两条虚线之间的垂直距离就是这个最优决策面对应的分类间隔。显然每一个可能把数据集正确分开的方向都有一个最优决策面（有些方向无论如何移动决策面的位置也不可能将两类样本完全分开），而不同方向的最优决策面的分类间隔通常是不同的，那个具有“最大间隔”的决策面就是SVM要寻找的最优解。而这个真正的最优解对应的两侧虚线所穿过的样本点，就是SVM中的支持样本点，称为"支持向量"。

#### 1、数学建模
求解这个"决策面"的过程，就是最优化。一个最优化问题通常有两个基本的因素：1）目标函数，也就是你希望什么东西的什么指标达到最好；2）优化对象，你期望通过改变哪些因素来使你的目标函数达到最优。在线性SVM算法中，目标函数显然就是那个"分类间隔"，而优化对象则是决策面。所以要对SVM问题进行数学建模，首先要对上述两个对象（"分类间隔"和"决策面"）进行数学描述。按照一般的思维习惯，我们先描述决策面。

数学建模的时候，先在二维空间建模，然后再推广到多维。

（1）"决策面"方程

我们都知道二维空间下一条直线的方式如下所示：

![](img/11.png)

现在我们做个小小的改变，让原来的x轴变成x1，y轴变成x2

![](img/12.png)

移项得：

![](img/13.png)

将公式向量化得：

![](img/14.png)

进一步向量化，用w列向量和x列向量和标量γ进一步向量化：

![](img/15.png)

其中，向量w和x分别为：

![](img/16.png)

这里w1=a，w2=-1。我们都知道，最初的那个直线方程a和b的几何意义，a表示直线的斜率，b表示截距，a决定了直线与x轴正方向的夹角，b决定了直线与y轴交点位置。那么向量化后的直线的w和r的几何意义是什么呢？

现在假设：

![](img/17.png)

可得：

![](img/18.png)

在坐标轴上画出直线和向量w：

![](img/19.png)

蓝色的线代表向量w，红色的线代表直线y。我们可以看到向量w和直线的关系为垂直关系。这说明了向量w也控制这直线的方向，只不过是与这个直线的方向是垂直的。标量γ的作用也没有变，依然决定了直线的截距。此时，我们称w为直线的法向量。

二维空间的直线方程已经推导完成，将其推广到n维空间，就变成了超平面方程。(一个超平面，在二维空间的例子就是一个直线)但是它的公式没变，依然是：

![](img/20.png)

不同之处在于：

![](img/21.png)

我们已经顺利推导出了"决策面"方程，它就是我们的超平面方程，之后，我们统称其为超平面方程。

（2）"分类间隔"方程

现在，我们依然对于一个二维平面的简单例子进行推导。

![](img/22.png)

我们已经知道间隔的大小实际上就是支持向量对应的样本点到决策面的距离的二倍。那么图中的距离d我们怎么求？我们高中都学过，点到直线的距离距离公式如下：

![](img/23.png)

公式中的直线方程为Ax0+By0+C=0，点P的坐标为(x0,y0)。

现在，将直线方程扩展到多维，求得我们现在的超平面方程，对公式进行如下变形：

![](img/24.png)

这个d就是"分类间隔"。其中||w||表示w的二范数，求所有元素的平方和，然后再开方。比如对于二维平面：

![](img/25.png)

那么，

![](img/26.png)

我们目的是为了找出一个分类效果好的超平面作为分类器。分类器的好坏的评定依据是分类间隔W=2d的大小，即分类间隔w越大，我们认为这个超平面的分类效果越好。此时，求解超平面的问题就变成了求解分类间隔W最大化的为题。W的最大化也就是d最大化的。

（3）约束条件

看起来，我们已经顺利获得了目标函数的数学形式。但是为了求解w的最大值。我们不得不面对如下问题：

我们如何判断超平面是否将样本点正确分类？

我们知道要求距离d的最大值，我们首先需要找到支持向量上的点，怎么在众多的点中选出支持向量上的点呢？

上述我们需要面对的问题就是约束条件，也就是说我们优化的变量d的取值范围受到了限制和约束。事实上约束条件一直是最优化问题里最让人头疼的东西。但既然我们已经知道了这些约束条件确实存在，就不得不用数学语言对他们进行描述。但SVM算法通过一些巧妙的小技巧，将这些约束条件融合到一个不等式里面。

这个二维平面上有两种点，我们分别对它们进行标记：

- 红颜色的圆点标记为1，我们人为规定其为正样本；
- 蓝颜色的五角星标记为-1，我们人为规定其为负样本。

对每个样本点xi加上一个类别标签yi：

![](img/27.png)

如果我们的超平面方程能够完全正确地对上图的样本点进行分类，就会满足下面的方程：

![](img/28.png)

如果我们要求再高一点，假设决策面正好处于间隔区域的中轴线上，并且相应的支持向量对应的样本点到决策面的距离为d，那么公式进一步写成：

![](img/29.png)

上述公式的解释就是，对于所有分类标签为1和-1样本点，它们到直线的距离都大于等于d(支持向量上的样本点到超平面的距离)。公式两边都除以d，就可以得到：

![](img/30.png)

其中，

![](img/31.png)

因为||w||和d都是标量。所以上述公式的两个矢量，依然描述一条直线的法向量和截距。

![](img/32.png)

上述两个公式，都是描述一条直线，数学模型代表的意义是一样的。现在，让我们对wd和γd重新起个名字，就叫它们w和γ。因此，我们就可以说："对于存在分类间隔的两类样本点，我们一定可以找到一些超平面，使其对于所有的样本点均满足下面的条件："

![](img/33.png)

上述方程即给出了SVM最优化问题的约束条件。这时候，可能有人会问了，为什么标记为1和-1呢？因为这样标记方便我们将上述方程变成如下形式：

![](img/34.png)

正是因为标签为1和-1，才方便我们将约束条件变成一个约束方程，从而方便我们的计算。

（4）线性SVM优化问题基本描述

现在整合一下思路，我们已经得到我们的目标函数：

![](img/35.png)

我们的优化目标是是d最大化。我们已经说过，我们是用支持向量上的样本点求解d的最大化的问题的。那么支持向量上的样本点有什么特点呢？

![](img/36.png)

你赞同这个观点吗？所有支持向量上的样本点，都满足如上公式。如果不赞同，请重看"分类间隔"方程推导过程。

现在我们就可以将我们的目标函数进一步化简：

![](img/37.png)

因为，我们只关心支持向量上的点。随后我们求解d的最大化问题变成了||w||的最小化问题。进而||w||的最小化问题等效于

![](img/38.png)

为什么要做这样的等效呢？这是为了在进行最优化的过程中对目标函数求导时比较方便，但这绝对不影响最优化问题最后的求解。我们将最终的目标函数和约束条件放在一起进行描述：

![](img/39.png)

这里n是样本点的总个数，缩写s.t.表示"Subject to"，是"服从某某条件"的意思。上述公式描述的是一个典型的不等式约束条件下的二次型函数优化问题，同时也是支持向量机的基本数学模型。

（5）求解准备

我们已经得到支持向量机的基本数学模型，接下来的问题就是如何根据数学模型，求得我们想要的最优解。在学习求解方法之前，我们得知道一点，想用我下面讲述的求解方法有一个前提，就是我们的目标函数必须是凸函数。理解凸函数，我们还要先明确另一个概念，凸集。在凸几何中，凸集(convex set)是在)凸组合下闭合的放射空间的子集。看一幅图可能更容易理解：

![](img/40.png)

左右量图都是一个集合。如果集合中任意2个元素连线上的点也在集合中，那么这个集合就是凸集。显然，上图中的左图是一个凸集，上图中的右图是一个非凸集。

凸函数的定义也是如此，其几何意义表示为函数任意两点连线上的值大于对应自变量处的函数值。若这里凸集C即某个区间L，那么，设函数f为定义在区间L上的函数，若对L上的任意两点x1，x2和任意的实数λ，λ属于(0,1)，总有：

![](img/41.png)

则函数f称为L上的凸函数，当且仅当其上镜图（在函数图像上方的点集）为一个凸集。再看一幅图，也许更容易理解：

![](img/42.png)

像上图这样的函数，它整体就是一个非凸函数，我们无法获得全局最优解的，只能获得局部最优解。比如红框内的部分，如果单独拿出来，它就是一个凸函数。对于我们的目标函数：

![](img/43.png)

很显然，它是一个凸函数。所以，可以使用我接下来讲述的方法求取最优解。

通常我们需要求解的最优化问题有如下几类：

无约束优化问题，可以写为：

![](img/44.png)

有等式约束的优化问题，可以写为：

![](img/45.png)

有不等式约束的优化问题，可以写为：

![](img/46.png)

对于第(a)类的优化问题，尝试使用的方法就是费马大定理(Fermat)，即使用求取函数f(x)的导数，然后令其为零，可以求得候选最优值，再在这些候选值中验证；如果是凸函数，可以保证是最优解。这也就是我们高中经常使用的求函数的极值的方法。

对于第(b)类的优化问题，常常使用的方法就是拉格朗日乘子法（Lagrange Multiplier) ，即把等式约束h_i(x)用一个系数与f(x)写为一个式子，称为拉格朗日函数，而系数称为拉格朗日乘子。通过拉格朗日函数对各个变量求导，令其为零，可以求得候选值集合，然后验证求得最优值。

对于第(c)类的优化问题，常常使用的方法就是KKT条件。同样地，我们把所有的等式、不等式约束与f(x)写为一个式子，也叫拉格朗日函数，系数也称拉格朗日乘子，通过一些条件，可以求出最优值的必要条件，这个条件称为KKT条件。

了解到这些，现在让我们再看一下我们的最优化问题：

![](img/47.png)

现在，我们的这个对优化问题属于哪一类？很显然，它属于第(c)类问题。在学习求解最优化问题之前，我们还要学习两个东西：拉格朗日函数和KKT条件。

（6）拉格朗日函数

首先，我们先要从宏观的视野上了解一下拉格朗日对偶问题出现的原因和背景。

我们知道我们要求解的是最小化问题，所以一个直观的想法是如果我能够构造一个函数，使得该函数在可行解区域内与原目标函数完全一致，而在可行解区域外的数值非常大，甚至是无穷大，那么这个没有约束条件的新目标函数的优化问题就与原来有约束条件的原始目标函数的优化问题是等价的问题。这就是使用拉格朗日方程的目的，它将约束条件放到目标函数中，从而将有约束优化问题转换为无约束优化问题。

随后，人们又发现，使用拉格朗日获得的函数，使用求导的方法求解依然困难。进而，需要对问题再进行一次转换，即使用一个数学技巧：拉格朗日对偶。

所以，显而易见的是，我们在拉格朗日优化我们的问题这个道路上，需要进行下面二个步骤：

将有约束的原始目标函数转换为无约束的新构造的拉格朗日目标函数
使用拉格朗日对偶性，将不易求解的优化问题转化为易求解的优化
下面，进行第一步：将有约束的原始目标函数转换为无约束的新构造的拉格朗日目标函数

公式变形如下：

![](img/48.png)

其中αi是拉格朗日乘子，αi大于等于0，是我们构造新目标函数时引入的系数变量(我们自己设置)。现在我们令：

![](img/49.png)

当样本点不满足约束条件时，即在可行解区域外：

![](img/50.png)

此时，我们将αi设置为正无穷，此时θ(w)显然也是正无穷。

当样本点满足约束条件时，即在可行解区域内：

![](img/51.png)

此时，显然θ(w)为原目标函数本身。我们将上述两种情况结合一下，就得到了新的目标函数：

![](img/52.png)

此时，再看我们的初衷，就是为了建立一个在可行解区域内与原目标函数相同，在可行解区域外函数值趋近于无穷大的新函数，现在我们做到了。

现在，我们的问题变成了求新目标函数的最小值，即：

![](img/53.png)

这里用p*表示这个问题的最优值，且和最初的问题是等价的。

接下来，我们进行第二步：将不易求解的优化问题转化为易求解的优化

我们看一下我们的新目标函数，先求最大值，再求最小值。这样的话，我们首先就要面对带有需要求解的参数w和b的方程，而αi又是不等式约束，这个求解过程不好做。所以，我们需要使用拉格朗日函数对偶性，将最小和最大的位置交换一下，这样就变成了：

![](img/54.png)

交换以后的新问题是原始问题的对偶问题，这个新问题的最优值用d*来表示。而且d*<=p*。我们关心的是d=p的时候，这才是我们要的解。需要什么条件才能让d=p呢？

首先必须满足这个优化问题是凸优化问题。

其次，需要满足KKT条件。

凸优化问题的定义是：求取最小值的目标函数为凸函数的一类优化问题。目标函数是凸函数我们已经知道，这个优化问题又是求最小值。所以我们的最优化问题就是凸优化问题。

接下里，就是探讨是否满足KKT条件了。

（7）KKT条件

我们已经使用拉格朗日函数对我们的目标函数进行了处理，生成了一个新的目标函数。通过一些条件，可以求出最优值的必要条件，这个条件就是接下来要说的KKT条件。一个最优化模型能够表示成下列标准形式：

![](img/55.png)

KKT条件的全称是Karush-Kuhn-Tucker条件，KKT条件是说最优值条件必须满足以下条件：

- 条件一：经过拉格朗日函数处理之后的新目标函数L(w,b,α)对x求导为零：
- 条件二：h(x) = 0；
- 条件三：α*g(x) = 0；

对于我们的优化问题：

![](img/56.png)

显然，条件二已经满足了。另外两个条件为啥也满足呢？

这里原谅我省略一系列证明步骤，感兴趣的可以移步这里：点我查看

这里已经给出了很好的解释。现在，凸优化问题和KKT都满足了，问题转换成了对偶问题。而求解这个对偶学习问题，可以分为三个步骤：首先要让L(w,b,α)关于w和b最小化，然后求对α的极大，最后利用SMO算法求解对偶问题中的拉格朗日乘子。现在，我们继续推导。

（8）对偶问题求解

第一步：

根据上述推导已知：

![](img/57.png)

首先固定α，要让L(w,b,α)关于w和b最小化，我们分别对w和b偏导数，令其等于0，即：

![](img/58.png)

将上述结果带回L(w,b,α)得到：

![](img/59.png)

从上面的最后一个式子，我们可以看出，此时的L(w,b,α)函数只含有一个变量，即αi。

第二步：

现在内侧的最小值求解完成，我们求解外侧的最大值，从上面的式子得到

![](img/60.png)

现在我们的优化问题变成了如上的形式。对于这个问题，我们有更高效的优化算法，即序列最小优化（SMO）算法。我们通过这个优化算法能得到α，再根据α，我们就可以求解出w和b，进而求得我们最初的目的：找到超平面，即"决策平面"。

总结一句话：我们为啥使出费这么大的力气进行推导？因为我们要将最初的原始问题，转换到可以使用SMO算法求解的问题，这是一种最流行的求解方法。

对于上述问题，我们就可以使用SMO算法进行求解了，但是，SMO算法又是什么呢？

#### 2、SMO算法

现在，我们已经得到了可以用SMO算法求解的目标函数，但是对于怎么编程实现SMO算法还是感觉无从下手。那么现在就聊聊如何使用SMO算法进行求解。

（1）Platt的SMO算法

1996年，John Platt发布了一个称为SMO的强大算法，用于训练SVM。SM表示序列最小化(Sequential Minimal Optimizaion)。Platt的SMO算法是将大优化问题分解为多个小优化问题来求解的。这些小优化问题往往很容易求解，并且对它们进行顺序求解的结果与将它们作为整体来求解的结果完全一致的。在结果完全相同的同时，SMO算法的求解时间短很多。

SMO算法的目标是求出一系列alpha和b，一旦求出了这些alpha，就很容易计算出权重向量w并得到分隔超平面。

SMO算法的工作原理是：每次循环中选择两个alpha进行优化处理。一旦找到了一对合适的alpha，那么就增大其中一个同时减小另一个。这里所谓的"合适"就是指两个alpha必须符合以下两个条件，条件之一就是两个alpha必须要在间隔边界之外，而且第二个条件则是这两个alpha还没有进行过区间化处理或者不在边界上。

（2）SMO算法的解法
先来定义特征到结果的输出函数为：

![](img/61.png)

接着，我们回忆一下原始优化问题，如下：

![](img/62.png)

求导得：

![](img/63.png)

将上述公式带入输出函数中：

![](img/64.png)

与此同时，拉格朗日对偶后得到最终的目标化函数：

![](img/65.png)

将目标函数变形，在前面增加一个符号，将最大值问题转换成最小值问题：

![](img/66.png)

实际上，对于上述目标函数，是存在一个假设的，即数据100%线性可分。但是，目前为止，我们知道几乎所有数据都不那么"干净"。这时我们就可以通过引入所谓的松弛变量(slack variable)，来允许有些数据点可以处于超平面的错误的一侧。这样我们的优化目标就能保持仍然不变，但是此时我们的约束条件有所改变：

![](img/67.png)

根据KKT条件可以得出其中αi取值的意义为：

![](img/68.png)

- 对于第1种情况，表明αi是正常分类，在边界内部；
- 对于第2种情况，表明αi是支持向量，在边界上；
- 对于第3种情况，表明αi是在两条边界之间。

而最优解需要满足KKT条件，即上述3个条件都得满足，以下几种情况出现将会不满足：

![](img/69.png)

也就是说，如果存在不能满足KKT条件的αi，那么需要更新这些αi，这是第一个约束条件。此外，更新的同时还要受到第二个约束条件的限制，即：

![](img/70.png)

因为这个条件，我们同时更新两个α值，因为只有成对更新，才能保证更新之后的值仍然满足和为0的约束，假设我们选择的两个乘子为α1和α2：

![](img/71.png)

其中， ksi为常数。因为两个因子不好同时求解，所以可以先求第二个乘子α2的解（α2 new），得到α2的解（α2 new）之后，再用α2的解（α2 new）表示α1的解（α1 new ）。为了求解α2 new ，得先确定α2 new的取值范围。假设它的上下边界分别为H和L，那么有：

![](img/72.png)

接下来，综合下面两个条件：

![](img/73.png)

当y1不等于y2时，即一个为正1，一个为负1的时候，可以得到：

![](img/74.png)

所以有：

![](img/75.png)

此时，取值范围如下图所示：

![](img/76.png)

当y1等于y2时，即两个都为正1或者都为负1，可以得到：

![](img/77.png)

所以有：

![](img/78.png)

此时，取值范围如下图所示：

![](img/79.png)

如此，根据y1和y2异号或同号，可以得出α2 new的上下界分别为：

![](img/80.png)

这个界限就是编程的时候需要用到的。已经确定了边界，接下来，就是推导迭代式，用于更新 α值。

我们已经知道，更新α的边界，接下来就是讨论如何更新α值。我们依然假设选择的两个乘子为α1和α2。固定这两个乘子，进行推导。于是目标函数变成了：

![](img/81.png)

为了描述方便，我们定义如下符号：

![](img/82.png)

最终目标函数变为：

![](img/83.png)

我们不关心constant的部分，因为对于α1和α2来说，它们都是常数项，在求导的时候，直接变为0。对于这个目标函数，如果对其求导，还有个未知数α1，所以要推导出α1和α2的关系，然后用α2代替α1，这样目标函数就剩一个未知数了，我们就可以求导了，推导出迭代公式。所以现在继续推导α1和α2的关系。注意第一个约束条件：

![](img/84.png)

我们在求α1和α2的时候，可以将α3,α4,...,αn和y3,y4,...,yn看作常数项。因此有：

![](img/85.png)

我们不必关心常数B的大小，现在将上述等式两边同时乘以y1，得到(y1y1=1)：

![](img/86.png)

其中γ为常数By1，我们不关心这个值，s=y1y2。接下来，我们将得到的α1带入W(α2)公式得：

![](img/87.png)

这样目标函数中就只剩下α2了，我们对其求偏导（注意：s=y1y2，所以s的平方为1，y1的平方和y2的平方均为1）：

![](img/88.png)

继续化简，将s=y1y2带入方程。

![](img/89.png)

我们令：

![](img/90.png)

Ei为误差项，η为学习速率。

再根据我们已知的公式：

![](img/91.png)

将α2 new继续化简得：

![](img/92.png)

这样，我们就得到了最终需要的迭代公式。这个是没有经过剪辑是的解，需要考虑约束：

![](img/93.png)

根据之前推导的α取值范围，我们得到最终的解析解为：

![](img/94.png)

又因为：

![](img/95.png)

消去γ得：

![](img/96.png)

这样，我们就知道了怎样计算α1和α2了，也就是如何对选择的α进行更新。

当我们更新了α1和α2之后，需要重新计算阈值b，因为b关系到了我们f(x)的计算，也就关系到了误差Ei的计算。

我们要根据α的取值范围，去更正b的值，使间隔最大化。当α1 new在0和C之间的时候，根据KKT条件可知，这个点是支持向量上的点。因此，满足下列公式：

![](img/97.png)

公式两边同时乘以y1得(y1y1=1)：

![](img/98.png)

因为我们是根据α1和α2的值去更新b，所以单独提出i=1和i=2的时候，整理可得：

![](img/99.png)

其中前两项为：

![](img/100.png)

将上述两个公式，整理得：

![](img/101.png)

同理可得b2 new为：

![](img/102.png)

当b1和b2都有效的时候，它们是相等的，即：

![](img/103.png)

当两个乘子都在边界上，则b阈值和KKT条件一致。当不满足的时候，SMO算法选择他们的中点作为新的阈值：

![](img/104.png)

最后，更新所有的α和b，这样模型就出来了，从而即可求出我们的分类函数。

现在，让我们梳理下SMO算法的步骤：

步骤1：计算误差：

![](img/105.png)

步骤2：计算上下界L和H：

![](img/106.png)

步骤3：计算η：

![](img/107.png)

步骤4：更新αj：

![](img/108.png)

步骤5：根据取值范围修剪αj：

![](img/109.png)

步骤6：更新αi：

![](img/110.png)

步骤7：更新b1和b2：

![](img/111.png)

步骤8：根据b1和b2更新b：

![](img/112.png)

## 四、编程求解线性SVM
---
已经梳理完了SMO算法实现步骤，接下来按照这个思路编写代码，进行实战练习。

#### 1、可视化数据集
我们先使用简单的数据集进行测试，数据集下载地址：[数据集下载](https://github.com/yaoguangju/machine_learning/blob/master/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/1.%E7%BA%BF%E6%80%A7SVM/testSet.txt)

编写程序可视化数据集，看下它是长什么样的：
```python
# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签

"""
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():                                     #逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])      #添加数据
        labelMat.append(float(lineArr[2]))                          #添加标签
    return dataMat,labelMat

"""
函数说明:数据可视化

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
Returns:
    无

"""
def showDataSet(dataMat, labelMat):
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)
```

运行程序，查看结果：

![](img/113.png)

这就是我们使用的二维数据集，显然线性可分。现在我们使用简化版的SMO算法进行求解。

#### 2、简化版SMO算法
按照上述已经推导的步骤编写代码：
```python
"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签

"""
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():                                     #逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])      #添加数据
        labelMat.append(float(lineArr[2]))                          #添加标签
    return dataMat,labelMat


"""
函数说明:随机选择alpha

Parameters:
    i - alpha
    m - alpha参数个数
Returns:
    j -

"""
def selectJrand(i, m):
    j = i                                 #选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j

"""
函数说明:修剪alpha

Parameters:
    aj - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    aj - alpah值

"""
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

"""
函数说明:简化版SMO算法

Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    无

"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #转换为numpy的mat存储
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    #初始化b参数，统计dataMatrix的维度
    b = 0; m,n = np.shape(dataMatrix)
    #初始化alpha参数，设为0
    alphas = np.mat(np.zeros((m,1)))
    #初始化迭代次数
    iter_num = 0
    #最多迭代matIter次
    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #步骤1：计算误差Ei
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            #优化alpha，更设定一定的容错率。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i,m)
                #步骤1：计算误差Ej
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                #保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                #步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                #步骤3：计算eta
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                #步骤4：更新alpha_j
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("alpha_j变化太小"); continue
                #步骤6：更新alpha_i
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #步骤7：更新b_1和b_2
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                #步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                #统计优化次数
                alphaPairsChanged += 1
                #打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num,i,alphaPairsChanged))
        #更新迭代次数
        if (alphaPairsChanged == 0): iter_num += 1
        else: iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b,alphas

"""
函数说明:分类结果可视化

Parameters:
    dataMat - 数据矩阵
    w - 直线法向量
    b - 直线解决
Returns:
    无

"""
def showClassifer(dataMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


"""
函数说明:计算w

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
    alphas - alphas值
Returns:
    无

"""
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)

```

程序运行结果：

![](img/114.png)

其中，中间的蓝线为求出来的分类器，用红圈圈出的点为支持向量点。

## 五、SMO算法优化
---
在几百个点组成的小规模数据集上，简化版SMO算法的运行是没有什么问题的，但是在更大的数据集上的运行速度就会变慢。简化版SMO算法的第二个α的选择是随机的，针对这一问题，我们可以使用启发式选择第二个α值，来达到优化效果。

#### 1、启发选择方式

下面这两个公式想必已经不再陌生：

![](img/115.png)

在实现SMO算法的时候，先计算η，再更新αj。为了加快第二个αj乘子的迭代速度，需要让直线的斜率增大，对于αj的更新公式，其中η值没有什么文章可做，于是只能令:

![](img/116.png)

因此，我们可以明确自己的优化方法了：

最外层循环，首先在样本中选择违反KKT条件的一个乘子作为最外层循环，然后用"启发式选择"选择另外一个乘子并进行这两个乘子的优化
在非边界乘子中寻找使得 |Ei - Ej| 最大的样本
如果没有找到，则从整个样本中随机选择一个样本
接下来，让我们看看完整版SMO算法如何实现。

#### 2、完整版SMO算法

完整版Platt SMO算法是通过一个外循环来选择违反KKT条件的一个乘子，并且其选择过程会在这两种方式之间进行交替：

- 在所有数据集上进行单遍扫描
- 在非边界α中实现单遍扫描
- 
非边界α指的就是那些不等于边界0或C的α值，并且跳过那些已知的不会改变的α值。所以我们要先建立这些α的列表，用于才能出α的更新状态。

在选择第一个α值后，算法会通过"启发选择方式"选择第二个α值。

#### 3、编写代码

我们首先构建一个仅包含init方法的optStruct类，将其作为一个数据结构来使用，方便我们对于重要数据的维护。代码思路和之前的简化版SMO算法是相似的，不同之处在于增加了优化方法，如果上篇文章已经看懂，我想这个代码会很好理解。创建一个svm-smo.py文件，编写代码如下：
```python
# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random


class optStruct:
	"""
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
	"""
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#根据矩阵行数初始化alpha参数为0	
		self.b = 0 										#初始化b参数为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。

def loadDataSet(fileName):
	"""
	读取数据
	Parameters:
	    fileName - 文件名
	Returns:
	    dataMat - 数据矩阵
	    labelMat - 数据标签
	"""
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     #逐行读取，滤除空格等
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])      #添加数据
		labelMat.append(float(lineArr[2]))                          #添加标签
	return dataMat,labelMat

def calcEk(oS, k):
	"""
	计算误差
	Parameters：
		oS - 数据结构
		k - 标号为k的数据
	Returns:
	    Ek - 标号为k的数据误差
	"""
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):
	"""
	函数说明:随机选择alpha_j的索引值

	Parameters:
	    i - alpha_i的索引值
	    m - alpha参数个数
	Returns:
	    j - alpha_j的索引值
	"""
	j = i                                 #选择一个不等于i的j
	while (j == i):
		j = int(random.uniform(0, m))
	return j

def selectJ(i, oS, Ei):
	"""
	内循环启发方式2
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
		Ei - 标号为i的数据误差
	Returns:
	    j, maxK - 标号为j或maxK的数据的索引值
	    Ej - 标号为j的数据误差
	"""
	maxK = -1; maxDeltaE = 0; Ej = 0 						#初始化
	oS.eCache[i] = [1,Ei]  									#根据Ei更新误差缓存
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]		#返回误差不为0的数据的索引值
	if (len(validEcacheList)) > 1:							#有不为0的误差
		for k in validEcacheList:   						#遍历,找到最大的Ek
			if k == i: continue 							#不计算i,浪费时间
			Ek = calcEk(oS, k)								#计算Ek
			deltaE = abs(Ei - Ek)							#计算|Ei-Ek|
			if (deltaE > maxDeltaE):						#找到maxDeltaE
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej										#返回maxK,Ej
	else:   												#没有不为0的误差
		j = selectJrand(i, oS.m)							#随机选择alpha_j的索引值
		Ej = calcEk(oS, j)									#计算Ej
	return j, Ej 											#j,Ej

def updateEk(oS, k):
	"""
	计算Ek,并更新误差缓存
	Parameters：
		oS - 数据结构
		k - 标号为k的数据的索引值
	Returns:
		无
	"""
	Ek = calcEk(oS, k)										#计算Ek
	oS.eCache[k] = [1,Ek]									#更新误差缓存


def clipAlpha(aj,H,L):
	"""
	修剪alpha_j
	Parameters:
	    aj - alpha_j的值
	    H - alpha上限
	    L - alpha下限
	Returns:
	    aj - 修剪后的alpah_j的值
	"""
	if aj > H: 
		aj = H
	if L > aj:
		aj = L
	return aj

def innerL(i, oS):
	"""
	优化的SMO算法
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
	Returns:
		1 - 有任意一对alpha值发生变化
		0 - 没有任意一对alpha值发生变化或变化太小
	"""
	#步骤1：计算误差Ei
	Ei = calcEk(oS, i)
	#优化alpha,设定一定的容错率。
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		#使用内循环启发方式2选择alpha_j,并计算Ej
		j,Ej = selectJ(i, oS, Ei)
		#保存更新前的aplpha值，使用深拷贝
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		#步骤2：计算上下界L和H
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H: 
			print("L==H")
			return 0
		#步骤3：计算eta
		eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
		if eta >= 0: 
			print("eta>=0")
			return 0
		#步骤4：更新alpha_j
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
		#步骤5：修剪alpha_j
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		#更新Ej至误差缓存
		updateEk(oS, j)
		if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
			print("alpha_j变化太小")
			return 0
		#步骤6：更新alpha_i
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		#更新Ei至误差缓存
		updateEk(oS, i)
		#步骤7：更新b_1和b_2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
		#步骤8：根据b_1和b_2更新b
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
	"""
	完整的线性SMO算法
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		maxIter - 最大迭代次数
	Returns:
		oS.b - SMO算法计算的b
		oS.alphas - SMO算法计算的alphas
	"""
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)					#初始化数据结构
	iter = 0 																						#初始化当前迭代次数
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):							#遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
		alphaPairsChanged = 0
		if entireSet:																				#遍历整个数据集   						
			for i in range(oS.m):        
				alphaPairsChanged += innerL(i,oS)													#使用优化的SMO算法
				print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		else: 																						#遍历非边界值
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet:																				#遍历一次后改为非边界遍历
			entireSet = False
		elif (alphaPairsChanged == 0):																#如果alpha没有更新,计算全样本遍历 
			entireSet = True  
		print("迭代次数: %d" % iter)
	return oS.b,oS.alphas 																			#返回SMO算法计算的b和alphas


def showClassifer(dataMat, classLabels, w, b):
	"""
	分类结果可视化
	Parameters:
		dataMat - 数据矩阵
	    w - 直线法向量
	    b - 直线解决
	Returns:
	    无
	"""
	#绘制样本点
	data_plus = []                                  #正样本
	data_minus = []                                 #负样本
	for i in range(len(dataMat)):
		if classLabels[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)              #转换为numpy矩阵
	data_minus_np = np.array(data_minus)            #转换为numpy矩阵
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
	#绘制直线
	x1 = max(dataMat)[0]
	x2 = min(dataMat)[0]
	a1, a2 = w
	b = float(b)
	a1 = float(a1[0])
	a2 = float(a2[0])
	y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
	plt.plot([x1, x2], [y1, y2])
	#找出支持向量点
	for i, alpha in enumerate(alphas):
		if alpha > 0:
			x, y = dataMat[i]
			plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
	plt.show()


def calcWs(alphas,dataArr,classLabels):
	"""
	计算w
	Parameters:
		dataArr - 数据矩阵
	    classLabels - 数据标签
	    alphas - alphas值
	Returns:
	    w - 计算得到的w
	"""
	X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
	m,n = np.shape(X)
	w = np.zeros((n,1))
	for i in range(m):
		w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w

if __name__ == '__main__':
	dataArr, classLabels = loadDataSet('testSet.txt')
	b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
	w = calcWs(alphas,dataArr, classLabels)
	showClassifer(dataArr, classLabels, w, b)

```

完整版SMO算法(左图)与简化版SMO算法(右图)运行结果对比如下图所示：

![](img/117.png)

图中画红圈的样本点为支持向量上的点，是满足算法的一种解。完整版SMO算法覆盖整个数据集进行计算，而简化版SMO算法是随机选择的。可以看出，完整版SMO算法选出的支持向量样点更多，更接近理想的分隔超平面。

对比两种算法的运算时间，我的测试结果是完整版SMO算法的速度比简化版SMO算法的速度快6倍左右。

其实，优化方法不仅仅是简单的启发式选择，还有其他优化方法，SMO算法速度还可以进一步提高。但是鉴于文章进度，这里不再进行展开。

## 六、非线性SVM
---
#### 1、核技巧

我们已经了解到，SVM如何处理线性可分的情况，而对于非线性的情况，SVM的处理方式就是选择一个核函数。简而言之：在线性不可分的情况下，SVM通过某种事先选择的非线性映射（核函数）将输入变量映到一个高维特征空间，将其变成在高维空间线性可分，在这个高维空间中构造最优分类超平面。

根据上篇文章，线性可分的情况下，可知最终的超平面方程为：

![](img/118.png)

将上述公式用内积来表示：

![](img/119.png)

对于线性不可分，我们使用一个非线性映射，将数据映射到特征空间，在特征空间中使用线性学习器，分类函数变形如下：

![](img/120.png)

其中ϕ从输入空间(X)到某个特征空间(F)的映射，这意味着建立非线性学习器分为两步：

- 首先使用一个非线性映射将数据变换到一个特征空间F；
- 然后在特征空间使用线性学习器分类。

如果有一种方法可以在特征空间中直接计算内积 <ϕ(xi),ϕ(x)>，就像在原始输入点的函数中一样，就有可能将两个步骤融合到一起建立一个分线性的学习器，这样直接计算的方法称为核函数方法。

这里直接给出一个定义：核是一个函数k，对所有x,z∈X，满足k(x,z)=<ϕ(xi),ϕ(x)>，这里ϕ(·)是从原始输入空间X到内积空间F的映射。

简而言之：如果不是用核技术，就会先计算线性映ϕ(x1)和ϕ(x2)，然后计算这它们的内积，使用了核技术之后，先把ϕ(x1)和ϕ(x2)的一般表达式<ϕ(x1),ϕ(x2)>=k(<ϕ(x1),ϕ(x2) >)计算出来，这里的<·，·>表示内积，k(·，·)就是对应的核函数，这个表达式往往非常简单，所以计算非常方便。

这种将内积替换成核函数的方式被称为核技巧(kernel trick)。

#### 2、非线性数据处理

已经知道了核技巧是什么，但是为什么要这样做呢？我们先举一个简单的例子，进行说明。假设二维平面x-y上存在若干点，其中点集A服从 {x,y|x^2+y^2=1}，点集B服从{x,y|x^2+y^2=9}，那么这些点在二维平面上的分布是这样的： 

![](img/121.png)

蓝色的是点集A，红色的是点集B，他们在xy平面上并不能线性可分，即用一条直线分割（ 虽然肉眼是可以识别的） 。采用映射(x,y)->(x,y,x^2+y^2)后，在三维空间的点的分布为：

![](img/122.png)

可见红色和蓝色的点被映射到了不同的平面，在更高维空间中是线性可分的（用一个平面去分割）。

上述例子中的样本点的分布遵循圆的分布。继续推广到椭圆的一般样本形式：

![](123.img/
可见红色和蓝色的点被映射到了不同的平面，在更高维空间中是线性可分的（用一个平面去分割）。

上述例子中的样本点的分布遵循圆的分布。继续推广到椭圆的一般样本形式：

![](img/124.png)

上图的两类数据分布为两个椭圆的形状，这样的数据本身就是不可分的。不难发现，这两个半径不同的椭圆是加上了少量的噪音生成得到的。所以，一个理想的分界应该也是一个椭圆，而不是一个直线。如果用X1和X2来表示这个二维平面的两个坐标的话，我们知道这个分界椭圆可以写为：

![](img/125.png)

这个方程就是高中学过的椭圆一般方程。注意上面的形式，如果我们构造另外一个五维的空间，其中五个坐标的值分别为：

![](img/126.png)

那么，显然我们可以将这个分界的椭圆方程写成如下形式：

![](img/127.png)

这个关于新的坐标Z1,Z2,Z3,Z4,Z5的方程，就是一个超平面方程，它的维度是5。也就是说，如果我们做一个映射 ϕ : 二维 → 五维，将 X1,X2按照上面的规则映射为 Z1,Z2,··· ,Z5，那么在新的空间中原来的数据将变成线性可分的，从而使用之前我们推导的线性分类算法就可以进行处理了。

我们举个简单的计算例子，现在假设已知的映射函数为：

![](img/128.png)

这个是一个从2维映射到5维的例子。如果没有使用核函数，根据上一小节的介绍，我们需要先结算映射后的结果，然后再进行内积运算。那么对于两个向量a1=(x1,x2)和a2=(y1,y2)有：

![](img/129.png)

另外，如果我们不进行映射计算，直接运算下面的公式：

![](img/130.png)

你会发现，这两个公式的计算结果是相同的。区别在于什么呢？

一个是根据映射函数，映射到高维空间中，然后再根据内积的公式进行计算，计算量大；
另一个则直接在原来的低维空间中进行计算，而不需要显式地写出映射后的结果，计算量小。
其实，在这个例子中，核函数就是：

![](img/131.png)

我们通过k(x1,x2)的低维运算得到了先映射再内积的高维运算的结果，这就是核函数的神奇之处，它有效减少了我们的计算量。在这个例子中，我们对一个2维空间做映射，选择的新的空间是原始空间的所以一阶和二阶的组合，得到了5维的新空间；如果原始空间是3维的，那么我们会得到19维的新空间，这个数目是呈爆炸性增长的。如果我们使用ϕ(·)做映射计算，难度非常大，而且如果遇到无穷维的情况，就根本无从计算了。所以使用核函数进行计算是非常有必要的。

#### 3、核技巧的实现

通过核技巧的转变，我们的分类函数变为：

![](img/132.png)

我们的对偶问题变成了：

![](img/133.png)

这样，我们就避开了高纬度空间中的计算。当然，我们刚刚的例子是非常简单的，我们可以手动构造出来对应映射的核函数出来，如果对于任意一个映射，要构造出对应的核函数就很困难了。因此，通常，人们会从一些常用的核函数中进行选择，根据问题和数据的不同，选择不同的参数，得到不同的核函数。接下来，要介绍的就是一个非常流行的核函数，那就是径向基核函数。

径向基核函数是SVM中常用的一个核函数。径向基核函数采用向量作为自变量的函数，能够基于向量举例运算输出一个标量。径向基核函数的高斯版本的公式如下：

![](img/134.png)

其中，σ是用户自定义的用于确定到达率(reach)或者说函数值跌落到0的速度参数。上述高斯核函数将数据从原始空间映射到无穷维空间。关于无穷维空间，我们不必太担心。高斯核函数只是一个常用的核函数，使用者并不需要确切地理解数据到底是如何表现的，而且使用高斯核函数还会得到一个理想的结果。如果σ选得很大的话，高次特征上的权重实际上衰减得非常快，所以实际上（数值上近似一下）相当于一个低维的子空间；反过来，如果σ选得很小，则可以将任意的数据映射为线性可分——当然，这并不一定是好事，因为随之而来的可能是非常严重的过拟合问题。不过，总的来说，通过调控参数σ，高斯核实际上具有相当高的灵活性，也是使用最广泛的核函数之一。

七、编程实现非线性SVM
---
接下来，我们将使用testSetRBF.txt和testSetRBF2.txt，前者作为训练集，后者作为测试集。数据集下载地址：[点我查看](https://github.com/yaoguangju/machine_learning/tree/master/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/2.%E9%9D%9E%E7%BA%BF%E6%80%A7SVM)

#### 1、可视化数据集
我们先编写程序简单看下数据集：

```python
# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('testSetRBF.txt')                        #加载训练集
    showDataSet(dataArr, labelArr)
```

程序运行结果：

![](img/135.png)

可见，数据明显是线性不可分的。下面我们根据公式，编写核函数，并增加初始化参数kTup用于存储核函数有关的信息，同时我们只要将之前的内积运算变成核函数的运算即可。最后编写testRbf()函数，用于测试。创建svmMLiA.py文件，编写代码如下：

```python
# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random


class optStruct:
	"""
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
	"""
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#根据矩阵行数初始化alpha参数为0	
		self.b = 0 										#初始化b参数为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
		self.K = np.mat(np.zeros((self.m,self.m)))		#初始化核K
		for i in range(self.m):							#计算所有数据的核K
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup): 
	"""
	通过核函数将数据转换更高维的空间
	Parameters：
		X - 数据矩阵
		A - 单个数据的向量
		kTup - 包含核函数信息的元组
	Returns:
	    K - 计算的核K
	"""
	m,n = np.shape(X)
	K = np.mat(np.zeros((m,1)))
	if kTup[0] == 'lin': K = X * A.T   					#线性核函数,只进行内积。
	elif kTup[0] == 'rbf': 								#高斯核函数,根据高斯核函数公式进行计算
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow*deltaRow.T
		K = np.exp(K/(-1*kTup[1]**2)) 					#计算高斯核K
	else: raise NameError('核函数无法识别')
	return K 											#返回计算的核K

def loadDataSet(fileName):
	"""
	读取数据
	Parameters:
	    fileName - 文件名
	Returns:
	    dataMat - 数据矩阵
	    labelMat - 数据标签
	"""
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     #逐行读取，滤除空格等
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])      #添加数据
		labelMat.append(float(lineArr[2]))                          #添加标签
	return dataMat,labelMat

def calcEk(oS, k):
	"""
	计算误差
	Parameters：
		oS - 数据结构
		k - 标号为k的数据
	Returns:
	    Ek - 标号为k的数据误差
	"""
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):
	"""
	函数说明:随机选择alpha_j的索引值

	Parameters:
	    i - alpha_i的索引值
	    m - alpha参数个数
	Returns:
	    j - alpha_j的索引值
	"""
	j = i                                 #选择一个不等于i的j
	while (j == i):
		j = int(random.uniform(0, m))
	return j

def selectJ(i, oS, Ei):
	"""
	内循环启发方式2
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
		Ei - 标号为i的数据误差
	Returns:
	    j, maxK - 标号为j或maxK的数据的索引值
	    Ej - 标号为j的数据误差
	"""
	maxK = -1; maxDeltaE = 0; Ej = 0 						#初始化
	oS.eCache[i] = [1,Ei]  									#根据Ei更新误差缓存
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]		#返回误差不为0的数据的索引值
	if (len(validEcacheList)) > 1:							#有不为0的误差
		for k in validEcacheList:   						#遍历,找到最大的Ek
			if k == i: continue 							#不计算i,浪费时间
			Ek = calcEk(oS, k)								#计算Ek
			deltaE = abs(Ei - Ek)							#计算|Ei-Ek|
			if (deltaE > maxDeltaE):						#找到maxDeltaE
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej										#返回maxK,Ej
	else:   												#没有不为0的误差
		j = selectJrand(i, oS.m)							#随机选择alpha_j的索引值
		Ej = calcEk(oS, j)									#计算Ej
	return j, Ej 											#j,Ej

def updateEk(oS, k):
	"""
	计算Ek,并更新误差缓存
	Parameters：
		oS - 数据结构
		k - 标号为k的数据的索引值
	Returns:
		无
	"""
	Ek = calcEk(oS, k)										#计算Ek
	oS.eCache[k] = [1,Ek]									#更新误差缓存


def clipAlpha(aj,H,L):
	"""
	修剪alpha_j
	Parameters:
	    aj - alpha_j的值
	    H - alpha上限
	    L - alpha下限
	Returns:
	    aj - 修剪后的alpah_j的值
	"""
	if aj > H: 
		aj = H
	if L > aj:
		aj = L
	return aj

def innerL(i, oS):
	"""
	优化的SMO算法
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
	Returns:
		1 - 有任意一对alpha值发生变化
		0 - 没有任意一对alpha值发生变化或变化太小
	"""
	#步骤1：计算误差Ei
	Ei = calcEk(oS, i)
	#优化alpha,设定一定的容错率。
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		#使用内循环启发方式2选择alpha_j,并计算Ej
		j,Ej = selectJ(i, oS, Ei)
		#保存更新前的aplpha值，使用深拷贝
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		#步骤2：计算上下界L和H
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H: 
			print("L==H")
			return 0
		#步骤3：计算eta
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta >= 0: 
			print("eta>=0")
			return 0
		#步骤4：更新alpha_j
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
		#步骤5：修剪alpha_j
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		#更新Ej至误差缓存
		updateEk(oS, j)
		if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
			print("alpha_j变化太小")
			return 0
		#步骤6：更新alpha_i
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		#更新Ei至误差缓存
		updateEk(oS, i)
		#步骤7：更新b_1和b_2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		#步骤8：根据b_1和b_2更新b
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
	"""
	完整的线性SMO算法
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		maxIter - 最大迭代次数
		kTup - 包含核函数信息的元组
	Returns:
		oS.b - SMO算法计算的b
		oS.alphas - SMO算法计算的alphas
	"""
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)				#初始化数据结构
	iter = 0 																						#初始化当前迭代次数
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):							#遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
		alphaPairsChanged = 0
		if entireSet:																				#遍历整个数据集   						
			for i in range(oS.m):        
				alphaPairsChanged += innerL(i,oS)													#使用优化的SMO算法
				print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		else: 																						#遍历非边界值
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet:																				#遍历一次后改为非边界遍历
			entireSet = False
		elif (alphaPairsChanged == 0):																#如果alpha没有更新,计算全样本遍历 
			entireSet = True  
		print("迭代次数: %d" % iter)
	return oS.b,oS.alphas 																			#返回SMO算法计算的b和alphas


def testRbf(k1 = 1.3):
	"""
	测试函数
	Parameters:
		k1 - 使用高斯核函数的时候表示到达率
	Returns:
	    无
	"""
	dataArr,labelArr = loadDataSet('testSetRBF.txt')						#加载训练集
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, ('rbf', k1))		#根据训练集计算b和alphas
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A > 0)[0]										#获得支持向量
	sVs = datMat[svInd] 													
	labelSV = labelMat[svInd];
	print("支持向量个数:%d" % np.shape(sVs)[0])
	m,n = np.shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))				#计算各个点的核
		predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 	#根据支持向量的点，计算超平面，返回预测结果
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1		#返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
	print("训练集错误率: %.2f%%" % ((float(errorCount)/m)*100)) 			#打印错误率
	dataArr,labelArr = loadDataSet('testSetRBF2.txt') 						#加载测试集
	errorCount = 0
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose() 		
	m,n = np.shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1)) 				#计算各个点的核			
		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b 		#根据支持向量的点，计算超平面，返回预测结果
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1    	#返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
	print("测试集错误率: %.2f%%" % ((float(errorCount)/m)*100)) 			#打印错误率


def showDataSet(dataMat, labelMat):
	"""
	数据可视化
	Parameters:
	    dataMat - 数据矩阵
	    labelMat - 数据标签
	Returns:
	    无
	"""
	data_plus = []                                  #正样本
	data_minus = []                                 #负样本
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)              #转换为numpy矩阵
	data_minus_np = np.array(data_minus)            #转换为numpy矩阵
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
	plt.show()

if __name__ == '__main__':
	testRbf()

```

运行结果如下图所示：

![](img/136.png)

可以看到，训练集错误率为1%，测试集错误率都是4%，训练耗时1.7s 。可以尝试更换不同的K1参数以观察测试错误率、训练错误率、支持向量个数随k1的变化情况。你会发现K1过大，会出现过拟合的情况，即训练集错误率低，但是测试集错误率高。

## 八、Sklearn构建SVM分类器
---

使用的数据集（testDigits和trainingDigits）：[点我查看](https://github.com/yaoguangju/machine_learning/tree/master/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/3.Sklearn%E6%9E%84%E5%BB%BASVM%E5%88%86%E7%B1%BB%E5%99%A8)

首先，我们先使用自己用python写的代码进行训练。创建文件svm-digits.py文件，编写代码如下：

```python
# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random



class optStruct:
	"""
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
	"""
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#根据矩阵行数初始化alpha参数为0
		self.b = 0 										#初始化b参数为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
		self.K = np.mat(np.zeros((self.m,self.m)))		#初始化核K
		for i in range(self.m):							#计算所有数据的核K
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup):
	"""
	通过核函数将数据转换更高维的空间
	Parameters：
		X - 数据矩阵
		A - 单个数据的向量
		kTup - 包含核函数信息的元组
	Returns:
	    K - 计算的核K
	"""
	m,n = np.shape(X)
	K = np.mat(np.zeros((m,1)))
	if kTup[0] == 'lin': K = X * A.T   					#线性核函数,只进行内积。
	elif kTup[0] == 'rbf': 								#高斯核函数,根据高斯核函数公式进行计算
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow*deltaRow.T
		K = np.exp(K/(-1*kTup[1]**2)) 					#计算高斯核K
	else: raise NameError('核函数无法识别')
	return K 											#返回计算的核K

def loadDataSet(fileName):
	"""
	读取数据
	Parameters:
	    fileName - 文件名
	Returns:
	    dataMat - 数据矩阵
	    labelMat - 数据标签
	"""
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     #逐行读取，滤除空格等
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])      #添加数据
		labelMat.append(float(lineArr[2]))                          #添加标签
	return dataMat,labelMat

def calcEk(oS, k):
	"""
	计算误差
	Parameters：
		oS - 数据结构
		k - 标号为k的数据
	Returns:
	    Ek - 标号为k的数据误差
	"""
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):
	"""
	函数说明:随机选择alpha_j的索引值

	Parameters:
	    i - alpha_i的索引值
	    m - alpha参数个数
	Returns:
	    j - alpha_j的索引值
	"""
	j = i                                 #选择一个不等于i的j
	while (j == i):
		j = int(random.uniform(0, m))
	return j

def selectJ(i, oS, Ei):
	"""
	内循环启发方式2
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
		Ei - 标号为i的数据误差
	Returns:
	    j, maxK - 标号为j或maxK的数据的索引值
	    Ej - 标号为j的数据误差
	"""
	maxK = -1; maxDeltaE = 0; Ej = 0 						#初始化
	oS.eCache[i] = [1,Ei]  									#根据Ei更新误差缓存
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]		#返回误差不为0的数据的索引值
	if (len(validEcacheList)) > 1:							#有不为0的误差
		for k in validEcacheList:   						#遍历,找到最大的Ek
			if k == i: continue 							#不计算i,浪费时间
			Ek = calcEk(oS, k)								#计算Ek
			deltaE = abs(Ei - Ek)							#计算|Ei-Ek|
			if (deltaE > maxDeltaE):						#找到maxDeltaE
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej										#返回maxK,Ej
	else:   												#没有不为0的误差
		j = selectJrand(i, oS.m)							#随机选择alpha_j的索引值
		Ej = calcEk(oS, j)									#计算Ej
	return j, Ej 											#j,Ej

def updateEk(oS, k):
	"""
	计算Ek,并更新误差缓存
	Parameters：
		oS - 数据结构
		k - 标号为k的数据的索引值
	Returns:
		无
	"""
	Ek = calcEk(oS, k)										#计算Ek
	oS.eCache[k] = [1,Ek]									#更新误差缓存


def clipAlpha(aj,H,L):
	"""
	修剪alpha_j
	Parameters:
	    aj - alpha_j的值
	    H - alpha上限
	    L - alpha下限
	Returns:
	    aj - 修剪后的alpah_j的值
	"""
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj

def innerL(i, oS):
	"""
	优化的SMO算法
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
	Returns:
		1 - 有任意一对alpha值发生变化
		0 - 没有任意一对alpha值发生变化或变化太小
	"""
	#步骤1：计算误差Ei
	Ei = calcEk(oS, i)
	#优化alpha,设定一定的容错率。
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		#使用内循环启发方式2选择alpha_j,并计算Ej
		j,Ej = selectJ(i, oS, Ei)
		#保存更新前的aplpha值，使用深拷贝
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		#步骤2：计算上下界L和H
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H:
			print("L==H")
			return 0
		#步骤3：计算eta
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta >= 0:
			print("eta>=0")
			return 0
		#步骤4：更新alpha_j
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
		#步骤5：修剪alpha_j
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		#更新Ej至误差缓存
		updateEk(oS, j)
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
			print("alpha_j变化太小")
			return 0
		#步骤6：更新alpha_i
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		#更新Ei至误差缓存
		updateEk(oS, i)
		#步骤7：更新b_1和b_2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		#步骤8：根据b_1和b_2更新b
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else:
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin',0)):
	"""
	完整的线性SMO算法
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		maxIter - 最大迭代次数
		kTup - 包含核函数信息的元组
	Returns:
		oS.b - SMO算法计算的b
		oS.alphas - SMO算法计算的alphas
	"""
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)				#初始化数据结构
	iter = 0 																						#初始化当前迭代次数
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):							#遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
		alphaPairsChanged = 0
		if entireSet:																				#遍历整个数据集
			for i in range(oS.m):
				alphaPairsChanged += innerL(i,oS)													#使用优化的SMO算法
				print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		else: 																						#遍历非边界值
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet:																				#遍历一次后改为非边界遍历
			entireSet = False
		elif (alphaPairsChanged == 0):																#如果alpha没有更新,计算全样本遍历
			entireSet = True
		print("迭代次数: %d" % iter)
	return oS.b,oS.alphas 																			#返回SMO算法计算的b和alphas


def img2vector(filename):
	"""
	将32x32的二进制图像转换为1x1024向量。
	Parameters:
		filename - 文件名
	Returns:
		returnVect - 返回的二进制图像的1x1024向量
	"""
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def loadImages(dirName):
	"""
	加载图片
	Parameters:
		dirName - 文件夹的名字
	Returns:
	    trainingMat - 数据矩阵
	    hwLabels - 数据标签
	"""
	from os import listdir
	hwLabels = []
	trainingFileList = listdir(dirName)
	m = len(trainingFileList)
	trainingMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		if classNumStr == 9: hwLabels.append(-1)
		else: hwLabels.append(1)
		trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
	return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
	"""
	测试函数
	Parameters:
		kTup - 包含核函数信息的元组
	Returns:
	    无
	"""
	dataArr,labelArr = loadImages('trainingDigits')
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A>0)[0]
	sVs=datMat[svInd]
	labelSV = labelMat[svInd];
	print("支持向量个数:%d" % np.shape(sVs)[0])
	m,n = np.shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print("训练集错误率: %.2f%%" % (float(errorCount)/m))
	dataArr,labelArr = loadImages('testDigits')
	errorCount = 0
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	m,n = np.shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print("测试集错误率: %.2f%%" % (float(errorCount)/m))

if __name__ == '__main__':
	testDigits()

```

SMO算法实现部分跟上文是一样的，我们新创建了img2vector()、loadImages()、testDigits()函数，它们分别用于二进制图形转换、图片加载、训练SVM分类器。我们自己的SVM分类器是个二类分类器，所以在设置标签的时候，将9作为负类，其余的0-8作为正类，进行训练。这是一种'ovr'思想，即one vs rest，就是对一个类别和剩余所有的类别进行分类。如果想实现10个数字的识别，一个简单的方法是，训练出10个分类器。这里简单起见，只训练了一个用于分类9和其余所有数字的分类器，运行结果如下：

![](img/137.png)

可以看到，虽然我们进行了所谓的"优化"，但是训练仍然很耗时，迭代10次，花费了5-6分钟左右。因为我们没有多进程、没有设置自动的终止条件，总之一句话，需要优化的地方太多了。尽管如此，我们训练后得到的结果还是不错的，可以看到训练集错误率为0，测试集错误率也仅为0.01%。

接下来，就是讲解本文的重头戏：sklearn.svm.SVC。

#### 1、sklearn.svm.SVC

官方英文文档手册：[点我查看](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

sklearn.svm模块提供了很多模型供我们使用，本文使用的是svm.SVC，它是基于libsvm实现的。

![](img/138.png)

让我们先看下SVC这个函数，一共有14个参数：

![](img/139.png)

参数说明如下：

- C：惩罚项，float类型，可选参数，默认为1.0，C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
- kernel：核函数类型，str类型，默认为'rbf'。可选参数为：
	- 'linear'：线性核函数
	- 'poly'：多项式核函数
	- 'rbf'：径像核函数/高斯核
	- 'sigmod'：sigmod核函数
	- 'precomputed'：核矩阵
	- precomputed表示自己提前计算好核函数矩阵，这时候算法内部就不再用核函数去计算核矩阵，而是直接用你给的核矩阵，核矩阵需要为n*n的。
- degree：多项式核函数的阶数，int类型，可选参数，默认为3。这个参数只对多项式核函数有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数。
- gamma：核函数系数，float类型，可选参数，默认为auto。只对'rbf' ,'poly' ,'sigmod'有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features。
- coef0：核函数中的独立项，float类型，可选参数，默认为0.0。只有对'poly' 和,'sigmod'核函数有用，是指其中的参数c。
- probability：是否启用概率估计，bool类型，可选参数，默认为False，这必须在调用fit()之前启用，并且会fit()方法速度变慢。
- shrinking：是否采用启发式收缩方式，bool类型，可选参数，默认为True。
- tol：svm停止训练的误差精度，float类型，可选参数，默认为1e^-3。
- cache_size：内存大小，float类型，可选参数，默认为200。指定训练所需要的内存，以MB为单位，默认为200MB。
- class_weight：类别权重，dict类型或str类型，可选参数，默认为None。给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。如果给定参数'balance'，则使用y的值自动调整与输入数据中的类频率成反比的权重。
- verbose：是否启用详细输出，bool类型，默认为False，此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
- max_iter：最大迭代次数，int类型，默认为-1，表示不限制。
- decision_function_shape：决策函数类型，可选参数'ovo'和'ovr'，默认为'ovr'。'ovo'表示one vs one，'ovr'表示one vs rest。
- random_state：数据洗牌时的种子值，int类型，可选参数，默认为None。伪随机数发生器的种子,在混洗数据时用于概率估计。
其实，只要自己写了SMO算法，每个参数的意思，大概都是能明白的。

2、编写代码
SVC很是强大，我们不用理解算法实现的具体细节，不用理解算法的优化方法。同时，它也满足我们的多分类需求。创建文件svm-svc.py文件，编写代码如下：
```python
# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC


def img2vector(filename):
	"""
	将32x32的二进制图像转换为1x1024向量。
	Parameters:
		filename - 文件名
	Returns:
		returnVect - 返回的二进制图像的1x1024向量
	"""
	#创建1x1024零向量
	returnVect = np.zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#按行读取
	for i in range(32):
		#读一行数据
		lineStr = fr.readline()
		#每一行的前32个元素依次添加到returnVect中
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	#返回转换后的1x1024向量
	return returnVect

def handwritingClassTest():
	"""
	手写数字分类测试
	Parameters:
		无
	Returns:
		无
	"""
	#测试集的Labels
	hwLabels = []
	#返回trainingDigits目录下的文件名
	trainingFileList = listdir('trainingDigits')
	#返回文件夹下文件的个数
	m = len(trainingFileList)
	#初始化训练的Mat矩阵,测试集
	trainingMat = np.zeros((m, 1024))
	#从文件名中解析出训练集的类别
	for i in range(m):
		#获得文件的名字
		fileNameStr = trainingFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#将获得的类别添加到hwLabels中
		hwLabels.append(classNumber)
		#将每一个文件的1x1024数据存储到trainingMat矩阵中
		trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
	clf = SVC(C=200,kernel='rbf')
	clf.fit(trainingMat,hwLabels)
	#返回testDigits目录下的文件列表
	testFileList = listdir('testDigits')
	#错误检测计数
	errorCount = 0.0
	#测试数据的数量
	mTest = len(testFileList)
	#从文件中解析出测试集的类别并进行分类测试
	for i in range(mTest):
		#获得文件的名字
		fileNameStr = testFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#获得测试集的1x1024向量,用于训练
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		#获得预测结果
		# classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		classifierResult = clf.predict(vectorUnderTest)
		print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
	handwritingClassTest()
```
运行结果如下：

![](img/140.png)

九、总结
---
#### 1、SVM的优缺点
**优点**

- 可用于线性/非线性分类，也可以用于回归，泛化错误率低，也就是说具有良好的学习能力，且学到的结果具有很好的推广性。
- 可以解决小样本情况下的机器学习问题，可以解决高维问题，可以避免神经网络结构选择和局部极小点问题。
- SVM是最好的现成的分类器，现成是指不加修改可直接使用。并且能够得到较低的错误率，SVM可以对训练集之外的数据点做很好的分类决策。

**缺点**

- 对参数调节和和函数的选择敏感。


#### 2、其他
至此，关于SVM的文章已经写完，还有一些理论和细节，可能会在今后的文章提及。

## 参考资料
---
- 本文中提到的手写数字识别，来自于《机器学习实战》的第六章支持向量机。
- 本文的理论部分，参考《机器学习实战》的第六章支持向量机。