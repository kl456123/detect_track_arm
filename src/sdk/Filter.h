#ifndef _SDK_FILTER_H_
#define _SDK_FILTER_H_

namespace indem {
    class CFilter {
    public:
        //��ʼ�������ü��ʱ��
        CFilter(int interval);
        void SetThreshold(int interval);
        //���õ�ǰʱ���
        bool IsPass(double now);

    private:
        int m_iInterval;
        double m_dLastTime;
    };
}

#endif