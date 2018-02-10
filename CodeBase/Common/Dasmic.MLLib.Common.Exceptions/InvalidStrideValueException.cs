using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{

    public class InvalidStrideValueException : Exception
    {
        public InvalidStrideValueException(
            string message,
            Exception innerException) : base(message, innerException)
            {

        }

        public InvalidStrideValueException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_cnn_stride_value;
            }
        }
    }
}
