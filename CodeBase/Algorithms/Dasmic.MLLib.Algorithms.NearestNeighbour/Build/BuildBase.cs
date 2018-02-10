using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NearestNeighbour
{
    public abstract class BuildBase: Common.MLCore.BuildBase
    {
        protected IDistanceMeasure _distanceMeasure;

        public override abstract Common.MLCore.ModelBase
            BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute);
      
    }
}
