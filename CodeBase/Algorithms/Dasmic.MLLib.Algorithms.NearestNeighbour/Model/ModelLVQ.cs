using System.Collections.Generic;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Math.Statistics;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    /// <summary>
    /// Learning Vector Quantization
    /// </summary>
    public class ModelLVQ: Dasmic.MLLib.Algorithms.NearestNeighbour.ModelBase
    {
        public ModelLVQ(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes,
                                double [][] data) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes,
                                    data)
        {
            _data = data;
            _k = 1;
        }


    }
}
