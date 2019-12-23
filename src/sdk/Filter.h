#ifndef _SDK_FILTER_H_
#define _SDK_FILTER_H_

namespace indem {
    class CFilter {
    public:
        //初始化并设置间隔时间
        CFilter(int interval);
        void SetThreshold(int interval);
        //设置当前时间戳
        bool IsPass(double now);

    private:
        int m_iInterval;
        double m_dLastTime;
    };
}

#endif