using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.Regression
{
    public abstract class ModelBase:Common.MLCore.ModelBase
    {
        public ModelBase(double missingValue,
                                int indexTargetAttribute, int countAttributes):
                                base(missingValue, indexTargetAttribute, countAttributes)
        {

        }

        public abstract override
            double RunModelForSingleData(double[] data);

        //Serialization Routine
        public abstract override void SaveModel(string filePath);
        //Deserialization Routine
        public abstract override void LoadModel(string filePath);

        
    }
}
