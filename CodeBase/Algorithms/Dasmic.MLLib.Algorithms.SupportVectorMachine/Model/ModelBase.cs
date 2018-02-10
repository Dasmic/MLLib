using Dasmic.MLLib.Common.MLCore;


namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    public abstract class ModelBase:MLLib.Common.MLCore.ModelBase
    {
        public ModelBase(double missingValue,
                                int indexTargetAttribute, int countAttributes) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {
          
        }

        //Serialization Routine
        public abstract override void SaveModel(string filePath);

        //Deserialization Routine
        public abstract override void LoadModel(string filePath);
        
    }
}
