using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction
{
    public class MaxPooling: IPoolingFunction
    {        
        private double _maxValue;

        public MaxPooling()
        {
            _maxValue = double.MinValue;
        }


        /// <summary>
        /// Returns the max values among values
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double GetValue()
        {
            return _maxValue;
        }

        public void AddValue(double value)
        {
            if (value > _maxValue)
                _maxValue = value;
        }
    }
}
