using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.InstanceBased
{
    public abstract class ModelBase : MLLib.Common.MLCore.ModelBase
    {
        public override abstract
           double RunModelForSingleData(double[] data);

        public ModelBase(double missingValue,
                               int indexTargetAttribute, int countAttributes) :
                               base(missingValue, indexTargetAttribute, countAttributes)
        {

        }

        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }
    }
}
