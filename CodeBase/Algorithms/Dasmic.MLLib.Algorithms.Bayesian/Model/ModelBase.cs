using System;
using System.Collections.Generic;

namespace Dasmic.MLLib.Algorithms.Bayesian
{
    public abstract class ModelBase : Common.MLCore.ModelBase
    {
        
        public ModelBase(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes)
        {
            
        }

        //Will work for both discrete and continuous values
        public abstract override double RunModelForSingleData(double[] data);
        
        //Serialization Routine
        public override  void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }
    }
}
