using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SingleConvolutionLayerInput : LayerBase
    {       
        private int _depth; //_depth is same as number of filters
        
        
        public SingleConvolutionLayerInput()
        {

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="data">3D array in form depth, columns, rows</param>
        public SingleConvolutionLayerInput(double[][][] data, 
                                            int padding)
        {
            _depth = data.Length; //For Input Layer this will be mostly 3 for each RGB
            
            SetupFilterUnits(_depth, data[0].Length,
                                    data[1].Length);           
            //TODO: Add Padding in Input Layer
        }

        private void SetupFilterUnits(int noFilterUnits,
                                      int noColumns,
                                      int noRows)
        {
            FilterUnits = new SingleFilterUnit[_depth];
            for (int ii = 0; ii < FilterUnits.Length; ii++)
                FilterUnits[ii] = new SingleFilterUnit( noColumns, noRows, 0, 0,
                                            MaxParallelThreads,WeightBaseValue);
        }
        

    }
}
