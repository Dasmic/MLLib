using System;
using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.DiscriminantAnalysis
{
    public abstract class ModelBase: Common.MLCore.ModelBase
    {

        public ModelBase(double missingValue,
                                int indexTargetAttribute, 
                                int countAttributes) :
                                base(missingValue, 
                                    indexTargetAttribute,
                                    countAttributes)
        {

        }

        public override abstract double RunModelForSingleData(double[] data);

        //Serialization Routine
        public override abstract void SaveModel(string filePath);

        //Deserialization Routine
        public override abstract void LoadModel(string filePath);
    }
}
