using System.Collections.Generic;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Math.Statistics;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    public class ModelkNN: Dasmic.MLLib.Algorithms.NearestNeighbour.ModelBase
    {
        public ModelkNN(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes,
                                double [][] data) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes,
                                    data)
        {
            _data = data;
            _k = 3;
        }

        public void setParameters(IDistanceMeasure distanceMeasure,
                                    int k)
        {
            if(distanceMeasure!=null)
                _distanceMeasure = distanceMeasure;
            if(k>0)
                _k = k;
        }

    }
}
