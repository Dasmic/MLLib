using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class ModelNotReadyException : Exception
    {
        public ModelNotReadyException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_model_not_ready;
            }
        }


    }
}

