using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class ModelRunException:Exception
    {

        public ModelRunException(string message, Exception innerException):base(message,innerException)
        {
            
        }

        public override string Message
        {
            get
            {
                
                return Resources.strings_messages.exception_model_run
                    +":" + base.Message;
            }
        }

       
    }

    
}
