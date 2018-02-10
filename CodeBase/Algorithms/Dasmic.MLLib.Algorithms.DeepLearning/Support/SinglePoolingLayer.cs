using Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SinglePoolingLayer:LayerBase
    {
        private IPoolingFunction _poolingFunction;

        public SinglePoolingLayer(LayerBase upStreamLayer,  //Number of Units in pool layer will be same 
                                       int sizeWindow, //Both Height and Width will be same
                                       int sizeStride)
        {
            _upstreamLayer = upStreamLayer;
            _poolingFunction = new MaxPooling();
            int noOfPoolingUnits = _upstreamLayer.GetNumberOfFilterUnits(); //No of pooling Units should equal upstream Filter neurons
        }
        
        private void SetupPoolingUnits(int noOfPoolingUnits,
                                        int sizeFilter,
                                        int sizeStride)
        {
            FilterUnits = new SinglePoolingUnit[noOfPoolingUnits];
            for (int ii = 0; ii < FilterUnits.Length; ii++)
            {
                FilterUnits[ii] = new SinglePoolingUnit(
                                        _upstreamLayer,
                                        sizeFilter,
                                        sizeStride,
                                        _maxParallelThreads);
            }
        }

        /// <summary>
        /// Default pooling function is MaxPool. Use this function to
        /// overwrite the default value
        /// </summary>
        /// <param name="poolingFunction"></param>
        public void SetPoolingFunction(IPoolingFunction poolingFunction)
        {
            _poolingFunction = poolingFunction;

            //Update activation in each filter
            foreach (SinglePoolingUnit spn in FilterUnits)
            {
                spn.SetPoolingFunction(_poolingFunction);
            }
        }
    }
}
