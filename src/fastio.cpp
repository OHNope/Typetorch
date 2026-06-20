module;

#include <fast_io.h>

module fastio;

namespace fastio_detail
{
    template <class T>
    void write_one(T const &value)
    {
        ::fast_io::details::print_after_io_print_forward<false>(
            ::fast_io::io_print_forward<char>(::fast_io::io_print_alias(value)));
    }
}

namespace fast_io
{
inline namespace io
{
    void write(char const *value, size_type size)
    {
        fastio_detail::write_one(::fast_io::mnp::strvw(value, value + size));
    }

    void write(char value)
    {
        fastio_detail::write_one(value);
    }

    void write(bool value)
    {
        fastio_detail::write_one(value);
    }

    void write(short value)
    {
        fastio_detail::write_one(value);
    }

    void write(unsigned short value)
    {
        fastio_detail::write_one(value);
    }

    void write(int value)
    {
        fastio_detail::write_one(value);
    }

    void write(unsigned int value)
    {
        fastio_detail::write_one(value);
    }

    void write(long value)
    {
        fastio_detail::write_one(value);
    }

    void write(unsigned long value)
    {
        fastio_detail::write_one(value);
    }

    void write(long long value)
    {
        fastio_detail::write_one(value);
    }

    void write(unsigned long long value)
    {
        fastio_detail::write_one(value);
    }

    void write(float value)
    {
        fastio_detail::write_one(value);
    }

    void write(double value)
    {
        fastio_detail::write_one(value);
    }

    void write(long double value)
    {
        fastio_detail::write_one(static_cast<double>(value));
    }
}
}
