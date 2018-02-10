using System;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public abstract class UnitBase
    {
        protected int _stride;
        protected int _maxParallelThreads;        
        protected LayerBase _upStreamLayer;

        //Known as ComputedValue in other NN        
        //Known as FilterMap in a Convolutional layer
        public double[][] ValueMap;  //Will be used by both Convolution and Pooling layers

        protected void SetValueMap(int windowSize)
        {
            int noOfInputColumns = _upStreamLayer.GetValueMapNoOfColumns();
            int noOfInputRows = _upStreamLayer.GetValueMapNoOfRows();
            SetValueMap(noOfInputColumns, noOfInputRows, windowSize);
        }

        protected void SetValueMap(int noOfInputColumns, 
                                    int noOfInputRows,
                                    int windowSize)
        {            
            //Computed value will depend on stride size
            int noOfOutputCols = noOfInputColumns;
            int noOfOutputRows = noOfInputRows;

            //For Input Layer ComputedValue/FilterMap values remain the same
            if (windowSize != 0) //If not an input layer
            {
                //Standard formula to determine FilterMap size
                double _temp = ((noOfInputColumns - windowSize)
                                        / _stride) + 1;
                if (Math.Round(_temp) != _temp)
                    throw new InvalidStrideValueException();
                else
                    noOfOutputCols = Convert.ToInt32(_temp);

                _temp = ((noOfInputRows - windowSize)
                                       / _stride) + 1;
                if (Math.Round(_temp) != _temp)
                    throw new InvalidStrideValueException();
                else
                    noOfOutputRows = Convert.ToInt32(_temp);
            }

            //Set FilterMap Size            
            ValueMap = SupportFunctions.Get2DArray(noOfOutputCols,
                                                        noOfOutputRows);
        }

        public UnitBase(LayerBase upStreamLayer,
                            int stride, 
                            int maxParallelThreads)
        {
            _stride = stride;
            _maxParallelThreads = maxParallelThreads;
            _upStreamLayer = upStreamLayer;
        }

        public int GetValueMapNoOfRows()
        {
            return ValueMap[0].Length;
        }

        public  int GetValueMapNoOfColumns()
        {
            return ValueMap.Length;
        }

        public abstract void ComputeValueMap();

    }
}
