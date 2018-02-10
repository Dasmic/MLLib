using Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SinglePoolingUnit : UnitBase
    {
        private IPoolingFunction _poolingFunction;
        private int _poolSize;

        public SinglePoolingUnit(LayerBase upStreamLayer,                                        
                                        int windowSize,
                                        int strideSize,
                                        int maxParallelThreads) : base(upStreamLayer, strideSize, maxParallelThreads)
        {
            SetValueMap(windowSize); //Sets the Output Values 2D array
            _poolSize = windowSize;
        }

        public void SetPoolingFunction(IPoolingFunction activationFunction)
        {
            _poolingFunction = activationFunction;
        }

        /// <summary>
        /// Computes the ValueMap using the provided Pooling function
        /// </summary>
        public override void ComputeValueMap()
        {
            int poolSize = _poolSize; //Filter is same as window
            //Slide the filter Window to compute value for each Cell in ValueMap (VM)                        
            for (int idxRowVM = 0; idxRowVM < GetValueMapNoOfRows(); idxRowVM++)
            {//idxRowVM
                for (int idxColVM = 0; idxColVM < GetValueMapNoOfColumns(); idxColVM++)
                { //idxColVM                    
                    int idxUpValueMapCol = 0, idxUpValueMapRow = 0;
                    //Set the filter window co-ordinates in the upstream value map
                    //relative to the current Value Map
                    int upMapLeftCol = idxColVM * _stride;
                    int upMapLeftRow = idxRowVM * _stride;
                    //TODO: Assign value
                    for (int idxFilter = 0; idxFilter < _upStreamLayer.GetNumberOfFilterUnits();
                                idxFilter++) //For each Filter Unit Upstream
                    {
                        SingleFilterUnit upFilterUnit =
                                            (SingleFilterUnit)_upStreamLayer.GetFilterUnit(idxFilter);

                        for (int idxPoolRow = 0; idxPoolRow < poolSize; idxPoolRow++)
                            for (int idxPoolCol = 0; idxPoolCol < poolSize; idxPoolCol++)
                            {
                                //Update value                               
                                idxUpValueMapCol = upMapLeftCol + idxPoolCol;
                                idxUpValueMapRow = upMapLeftRow + idxPoolRow;
                                //Multiply upStream FilterMap by current Filter weights
                                _poolingFunction.AddValue(upFilterUnit.GetValueMapAtIndex(idxUpValueMapCol,
                                                                    idxUpValueMapRow));
                            } //idxFilterCol

                            //Compute Value
                    }//idxFilter
                    
                    //Apply pooling function - this saves time                    
                    ValueMap[idxColVM][idxRowVM] = 
                        _poolingFunction.GetValue(); ; //Assign value to FilterMap
                } //idxColFM
            } //idxRowFM  
        }
    }
}
