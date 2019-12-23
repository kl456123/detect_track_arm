#include "Filter.h"

namespace indem {

    CFilter::CFilter(int interval)
        :m_iInterval(interval)
        , m_dLastTime(0)
    {

    }

    void CFilter::SetThreshold(int interval)
    {
        m_iInterval = interval;
    }

    bool CFilter::IsPass(double now)
    {
        if (now - m_dLastTime > (m_iInterval - 1) / 1000.0f) {
            m_dLastTime = now;
            return true;
        }
        return false;
    }

}