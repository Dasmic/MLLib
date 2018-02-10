using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    class ModelID3 : ModelBase
    {
        public ModelID3(double missingValue,
                                int indexTargetAttribute,
                                int countAttributes) :
                                base(missingValue,
                                    indexTargetAttribute,
                                    countAttributes)
        { }
    }
}
