#pragma once
#include <atomic>
#include <mutex>
namespace indem {

    template<typename _Ty, typename _Mutex=std::mutex,typename _Lock=std::lock_guard<_Mutex> >
    class DoubleBuffer {
    public:
        DoubleBuffer() 
            :m_iReadIndex(0), m_iWriteIndex(1), m_bRead(true)
        {
            m_tBuffer[0] = _Ty();
            m_tBuffer[1] = _Ty();
        }

        DoubleBuffer(_Ty*& buff1, _Ty*& buff2)
            :m_iReadIndex(0), m_iWriteIndex(1) 
        {
            m_tBuffer[0] = buff1;
            m_tBuffer[1] = buff2;
        }
        void Write(const _Ty& val) {
            m_tBuffer[m_iWriteIndex]=val;
            m_bRead = false;
            if (m_mtx.try_lock()) {
                std::swap(m_iWriteIndex, m_iReadIndex);
                m_mtx.unlock();
            }
            //_Lock l(m_mtx);
        }

        _Ty Read() {
            _Lock l(m_mtx);
            m_bRead = true;
            return m_tBuffer[m_iReadIndex];
        }

        bool HasRead() {
            return m_bRead;
        }
    private:
        _Mutex m_mtx;
        _Ty m_tBuffer[2];               //ʹ��2����������������,һ����һ��д
        int m_iReadIndex;               //ָʾ��ǰ�Ķ�������
        int m_iWriteIndex;              //ָʾ��ǰ��д������
        std::atomic<bool> m_bRead;      //ָʾ�Ƿ��Ѷ���
        //std::atomic<bool> m_bRead;      //ָʾ�Ƿ��Ѷ���
    };
}