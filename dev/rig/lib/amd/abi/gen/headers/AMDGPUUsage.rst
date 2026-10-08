.. Excerpt of https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-20.1.0/llvm/docs/AMDGPUUsage.rst.
.. LLVM is under the Apache License v2.0 with LLVM Exceptions (SPDX:
.. Apache-2.0 WITH LLVM-exception).

  .. table:: AMDGPU ``EF_AMDGPU_MACH`` Values
     :name: amdgpu-ef-amdgpu-mach-table

     ========================================== ========== =============================
     Name                                       Value      Description (see
                                                           :ref:`amdgpu-processor-table`)
     ========================================== ========== =============================
     ``EF_AMDGPU_MACH_NONE``                    0x000      *not specified*
     ``EF_AMDGPU_MACH_R600_R600``               0x001      ``r600``
     ``EF_AMDGPU_MACH_R600_R630``               0x002      ``r630``
     ``EF_AMDGPU_MACH_R600_RS880``              0x003      ``rs880``
     ``EF_AMDGPU_MACH_R600_RV670``              0x004      ``rv670``
     ``EF_AMDGPU_MACH_R600_RV710``              0x005      ``rv710``
     ``EF_AMDGPU_MACH_R600_RV730``              0x006      ``rv730``
     ``EF_AMDGPU_MACH_R600_RV770``              0x007      ``rv770``
     ``EF_AMDGPU_MACH_R600_CEDAR``              0x008      ``cedar``
     ``EF_AMDGPU_MACH_R600_CYPRESS``            0x009      ``cypress``
     ``EF_AMDGPU_MACH_R600_JUNIPER``            0x00a      ``juniper``
     ``EF_AMDGPU_MACH_R600_REDWOOD``            0x00b      ``redwood``
     ``EF_AMDGPU_MACH_R600_SUMO``               0x00c      ``sumo``
     ``EF_AMDGPU_MACH_R600_BARTS``              0x00d      ``barts``
     ``EF_AMDGPU_MACH_R600_CAICOS``             0x00e      ``caicos``
     ``EF_AMDGPU_MACH_R600_CAYMAN``             0x00f      ``cayman``
     ``EF_AMDGPU_MACH_R600_TURKS``              0x010      ``turks``
     *reserved*                                 0x011 -    Reserved for ``r600``
                                                0x01f      architecture processors.
     ``EF_AMDGPU_MACH_AMDGCN_GFX600``           0x020      ``gfx600``
     ``EF_AMDGPU_MACH_AMDGCN_GFX601``           0x021      ``gfx601``
     ``EF_AMDGPU_MACH_AMDGCN_GFX700``           0x022      ``gfx700``
     ``EF_AMDGPU_MACH_AMDGCN_GFX701``           0x023      ``gfx701``
     ``EF_AMDGPU_MACH_AMDGCN_GFX702``           0x024      ``gfx702``
     ``EF_AMDGPU_MACH_AMDGCN_GFX703``           0x025      ``gfx703``
     ``EF_AMDGPU_MACH_AMDGCN_GFX704``           0x026      ``gfx704``
     *reserved*                                 0x027      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX801``           0x028      ``gfx801``
     ``EF_AMDGPU_MACH_AMDGCN_GFX802``           0x029      ``gfx802``
     ``EF_AMDGPU_MACH_AMDGCN_GFX803``           0x02a      ``gfx803``
     ``EF_AMDGPU_MACH_AMDGCN_GFX810``           0x02b      ``gfx810``
     ``EF_AMDGPU_MACH_AMDGCN_GFX900``           0x02c      ``gfx900``
     ``EF_AMDGPU_MACH_AMDGCN_GFX902``           0x02d      ``gfx902``
     ``EF_AMDGPU_MACH_AMDGCN_GFX904``           0x02e      ``gfx904``
     ``EF_AMDGPU_MACH_AMDGCN_GFX906``           0x02f      ``gfx906``
     ``EF_AMDGPU_MACH_AMDGCN_GFX908``           0x030      ``gfx908``
     ``EF_AMDGPU_MACH_AMDGCN_GFX909``           0x031      ``gfx909``
     ``EF_AMDGPU_MACH_AMDGCN_GFX90C``           0x032      ``gfx90c``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1010``          0x033      ``gfx1010``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1011``          0x034      ``gfx1011``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1012``          0x035      ``gfx1012``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1030``          0x036      ``gfx1030``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1031``          0x037      ``gfx1031``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1032``          0x038      ``gfx1032``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1033``          0x039      ``gfx1033``
     ``EF_AMDGPU_MACH_AMDGCN_GFX602``           0x03a      ``gfx602``
     ``EF_AMDGPU_MACH_AMDGCN_GFX705``           0x03b      ``gfx705``
     ``EF_AMDGPU_MACH_AMDGCN_GFX805``           0x03c      ``gfx805``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1035``          0x03d      ``gfx1035``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1034``          0x03e      ``gfx1034``
     ``EF_AMDGPU_MACH_AMDGCN_GFX90A``           0x03f      ``gfx90a``
     ``EF_AMDGPU_MACH_AMDGCN_GFX940``           0x040      ``gfx940``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1100``          0x041      ``gfx1100``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1013``          0x042      ``gfx1013``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1150``          0x043      ``gfx1150``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1103``          0x044      ``gfx1103``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1036``          0x045      ``gfx1036``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1101``          0x046      ``gfx1101``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1102``          0x047      ``gfx1102``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1200``          0x048      ``gfx1200``
     *reserved*                                 0x049      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX1151``          0x04a      ``gfx1151``
     ``EF_AMDGPU_MACH_AMDGCN_GFX941``           0x04b      ``gfx941``
     ``EF_AMDGPU_MACH_AMDGCN_GFX942``           0x04c      ``gfx942``
     *reserved*                                 0x04d      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX1201``          0x04e      ``gfx1201``
     ``EF_AMDGPU_MACH_AMDGCN_GFX950``           0x04f      ``gfx950``
     *reserved*                                 0x050      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC``     0x051      ``gfx9-generic``
     ``EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC``  0x052      ``gfx10-1-generic``
     ``EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC``  0x053      ``gfx10-3-generic``
     ``EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC``    0x054      ``gfx11-generic``
     ``EF_AMDGPU_MACH_AMDGCN_GFX1152``          0x055      ``gfx1152``.
     *reserved*                                 0x056      Reserved.
     *reserved*                                 0x057      Reserved.
     ``EF_AMDGPU_MACH_AMDGCN_GFX1153``          0x058      ``gfx1153``.
     ``EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC``    0x059      ``gfx12-generic``
     ``EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC``   0x05f      ``gfx9-4-generic``
     ========================================== ========== =============================

  .. table:: AMDGPU Generic Processors
     :name: amdgpu-generic-processor-table

     ==================== ============== ================= ================== ================= =================================
     Processor             Target        Supported         Target Features    Target Properties Target Restrictions
                           Triple        Processors        Supported
                           Architecture

     ==================== ============== ================= ================== ================= =================================
     ``gfx9-generic``     ``amdgcn``     - ``gfx900``      - xnack            - Absolute flat   - ``v_mad_mix`` instructions
                                         - ``gfx902``                           scratch           are not available on
                                         - ``gfx904``                                             ``gfx900``, ``gfx902``,
                                         - ``gfx906``                                             ``gfx909``, ``gfx90c``
                                         - ``gfx909``                                           - ``v_fma_mix`` instructions
                                         - ``gfx90c``                                             are not available on ``gfx904``
                                                                                                - sramecc is not available on
                                                                                                  ``gfx906``
                                                                                                - The following instructions
                                                                                                  are not available on ``gfx906``:

                                                                                                  - ``v_fmac_f32``
                                                                                                  - ``v_xnor_b32``
                                                                                                  - ``v_dot4_i32_i8``
                                                                                                  - ``v_dot8_i32_i4``
                                                                                                  - ``v_dot2_i32_i16``
                                                                                                  - ``v_dot2_u32_u16``
                                                                                                  - ``v_dot4_u32_u8``
                                                                                                  - ``v_dot8_u32_u4``
                                                                                                  - ``v_dot2_f32_f16``


     ``gfx9-4-generic``   ``amdgcn``     - ``gfx940``      - xnack            - Absolute flat   FP8 and BF8 instructions,
                                         - ``gfx941``      - sramecc            scratch         FP8 and BF8 conversion instructions,
                                         - ``gfx942``                                           as well as instructions with XF32 format support
                                         - ``gfx950``                                           are not available.


     ``gfx10-1-generic``  ``amdgcn``     - ``gfx1010``     - xnack            - Absolute flat   - The following instructions are
                                         - ``gfx1011``     - wavefrontsize64    scratch           not available on ``gfx1011``
                                         - ``gfx1012``     - cumode                               and ``gfx1012``
                                         - ``gfx1013``
                                                                                                  - ``v_dot4_i32_i8``
                                                                                                  - ``v_dot8_i32_i4``
                                                                                                  - ``v_dot2_i32_i16``
                                                                                                  - ``v_dot2_u32_u16``
                                                                                                  - ``v_dot2c_f32_f16``
                                                                                                  - ``v_dot4c_i32_i8``
                                                                                                  - ``v_dot4_u32_u8``
                                                                                                  - ``v_dot8_u32_u4``
                                                                                                  - ``v_dot2_f32_f16``

                                                                                                - BVH Ray Tracing instructions
                                                                                                  are not available on
                                                                                                  ``gfx1013``


     ``gfx10-3-generic``  ``amdgcn``     - ``gfx1030``     - wavefrontsize64  - Absolute flat   No restrictions.
                                         - ``gfx1031``     - cumode             scratch
                                         - ``gfx1032``
                                         - ``gfx1033``
                                         - ``gfx1034``
                                         - ``gfx1035``
                                         - ``gfx1036``


     ``gfx11-generic``    ``amdgcn``     - ``gfx1100``     - wavefrontsize64  - Architected     Various codegen pessimizations
                                         - ``gfx1101``     - cumode             flat scratch    are applied to work around some
                                         - ``gfx1102``                        - Packed          hazards specific to some targets
                                         - ``gfx1103``                          work-item       within this family.
                                         - ``gfx1150``                          IDs
                                         - ``gfx1151``
                                         - ``gfx1152``
                                         - ``gfx1153``                                          Not all VGPRs can be used on:

                                                                                                - ``gfx1100``
                                                                                                - ``gfx1101``
                                                                                                - ``gfx1151``

                                                                                                SALU floating point instructions
                                                                                                are not available on:

                                                                                                - ``gfx1150``
                                                                                                - ``gfx1151``
                                                                                                - ``gfx1152``
                                                                                                - ``gfx1153``

                                                                                                SGPRs are not supported for src1
                                                                                                in dpp instructions for:

                                                                                                - ``gfx1150``
                                                                                                - ``gfx1151``
                                                                                                - ``gfx1152``
                                                                                                - ``gfx1153``


     ``gfx12-generic``    ``amdgcn``     - ``gfx1200``     - wavefrontsize64  - Architected     No restrictions.
                                         - ``gfx1201``     - cumode             flat scratch
                                                                              - Packed
                                                                                work-item
                                                                                IDs
     ==================== ============== ================= ================== ================= =================================
