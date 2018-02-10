using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.InstanceBased
{
    public abstract  class BuildBase : MLLib.Common.MLCore.BuildBase
    {
        protected long _maxEpoch;
        protected double _learningRate;
        protected double _weightBaseValue;

        public BuildBase()
        {
            _maxEpoch = 500;
            _learningRate = .5;
        }

        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                           string[] attributeHeaders,
                           int indexTargetAttribute);
    }
}
