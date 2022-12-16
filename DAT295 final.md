我们希望改进原有基于多数投票的颜色选择机制，首先需要进行的就是选择更加精确的道路范围。例如，我们行驶在南北向的道路上，但是在目标区域内存在东西向的道路。这时如果使用多数投票对道路交通信息进行提取，会受到东西向道路交通状况的影响。因此我们希望设计一种算法排除与我们行驶方向无关的道路，减少其他道路对我们的影响。

卷积是一种非常适合这种情形的算法，由于我们的道路特征在图像中的区域是不固定的，同时我们采用的GPS轨迹坐标也是不精确的，我们无法使用经纬度坐标在图上对应点的颜色作为结果。卷积可以很好的处理这种特征不固定，需要我们进行大范围搜索的情形。

我们知道，车辆行驶的轨迹一定与道路的形态近似，因此我们可以使用车辆行驶的轨迹坐标构成一个卷积核作为目标特征，对目标区域进行搜索，获得高响应区域，这些区域的特征将与我们寻找的目标特征高度匹配，也被称为“interest area“。紧接着我们截取车辆当前时间点所在位置的一小块区域，在感兴趣的区域中统计多数颜色作为我们的最终结果。

We want to improve the original majority voting based color selection mechanism, and the first thing we need to do is to select a more precise road range. For example, we are driving on a north-south road, but an east-west road exists in the target area. At this point, if we use majority voting for road traffic information extraction, it will be affected by the east-west road traffic condition. Therefore, we want to design an algorithm to exclude roads that are not related to our driving direction and reduce the influence of other roads on us.

Convolution is a very suitable algorithm for this situation. Since our road features are not fixed in the area of the image, and also the GPS track coordinates we use are not precise, we cannot use the color of the corresponding point on the graph with latitude and longitude coordinates as the result. Convolution can handle this situation well where the features are not fixed and we need to do a wide range of search.

We know that the trajectory of a vehicle must be close to the shape of the road, so we can use the coordinates of the trajectory of the vehicle to form a convolution kernel as the target feature, and search the target area to get high response areas, the features of these areas will be highly matched with the target features we are looking for, also known as "interest area ". We then intercept a small area where the vehicle is located at the current point in time, and count the majority of colors in the area of interest as our final result.

在我们获得了感兴趣的道路之后，我们又面临了新的问题。通常情况下，道路具有两种方向的车道，并且两个方向上的道路交通状况可能存在差异。因此我们需要使用轨迹的方向来判断当前我们究竟是在哪一个车道上行驶，在感兴趣道路中排除方向不同的车道。

我们使用的方法是对不同方向的车道在卷积核中赋予不同的权重，这样在进行卷积操作后会产生不同的响应值，与我们当前行驶方向相同的道路会具有更高的响应。首先我们仍然按照原有的方法构建初始卷积核，然后计算当前行驶轨迹的方向向量，我们将卷积核中每一个特征点都向方向向量逆时针旋转九十度的方向平移x个像素（通常为6或11，分别对应不同的道路宽度）对应位置即为方向相反的车道。我们将当前行驶方向的车道的权重设置为2，对向车道的权重设置为1，这样可以保证我们能够探测到两个方向的道路并且有所区别。