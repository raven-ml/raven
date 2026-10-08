/*
 * Copyright 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

enum psp_gfx_fw_type {
	GFX_FW_TYPE_NONE        = 0,    /* */
	GFX_FW_TYPE_CP_ME       = 1,    /* CP-ME                    VG + RV */
	GFX_FW_TYPE_CP_PFP      = 2,    /* CP-PFP                   VG + RV */
	GFX_FW_TYPE_CP_CE       = 3,    /* CP-CE                    VG + RV */
	GFX_FW_TYPE_CP_MEC      = 4,    /* CP-MEC FW                VG + RV */
	GFX_FW_TYPE_CP_MEC_ME1  = 5,    /* CP-MEC Jump Table 1      VG + RV */
	GFX_FW_TYPE_CP_MEC_ME2  = 6,    /* CP-MEC Jump Table 2      VG      */
	GFX_FW_TYPE_RLC_V       = 7,    /* RLC-V                    VG      */
	GFX_FW_TYPE_RLC_G       = 8,    /* RLC-G                    VG + RV */
	GFX_FW_TYPE_SDMA0       = 9,    /* SDMA0                    VG + RV */
	GFX_FW_TYPE_SDMA1       = 10,   /* SDMA1                    VG      */
	GFX_FW_TYPE_DMCU_ERAM   = 11,   /* DMCU-ERAM                VG + RV */
	GFX_FW_TYPE_DMCU_ISR    = 12,   /* DMCU-ISR                 VG + RV */
	GFX_FW_TYPE_VCN         = 13,   /* VCN                           RV */
	GFX_FW_TYPE_UVD         = 14,   /* UVD                      VG      */
	GFX_FW_TYPE_VCE         = 15,   /* VCE                      VG      */
	GFX_FW_TYPE_ISP         = 16,   /* ISP                           RV */
	GFX_FW_TYPE_ACP         = 17,   /* ACP                           RV */
	GFX_FW_TYPE_SMU         = 18,   /* SMU                      VG      */
	GFX_FW_TYPE_MMSCH       = 19,   /* MMSCH                    VG      */
	GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM        = 20,   /* RLC GPM                  VG + RV */
	GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM        = 21,   /* RLC SRM                  VG + RV */
	GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL       = 22,   /* RLC CNTL                 VG + RV */
	GFX_FW_TYPE_UVD1        = 23,   /* UVD1                     VG-20   */
	GFX_FW_TYPE_TOC         = 24,   /* TOC                      NV-10   */
	GFX_FW_TYPE_RLC_P                           = 25,   /* RLC P                    NV      */
	GFX_FW_TYPE_RLC_IRAM                        = 26,   /* RLC_IRAM                 NV      */
	GFX_FW_TYPE_GLOBAL_TAP_DELAYS               = 27,   /* GLOBAL TAP DELAYS        NV      */
	GFX_FW_TYPE_SE0_TAP_DELAYS                  = 28,   /* SE0 TAP DELAYS           NV      */
	GFX_FW_TYPE_SE1_TAP_DELAYS                  = 29,   /* SE1 TAP DELAYS           NV      */
	GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS      = 30,   /* GLOBAL SE0/1 SKEW DELAYS NV      */
	GFX_FW_TYPE_SDMA0_JT                        = 31,   /* SDMA0 JT                 NV      */
	GFX_FW_TYPE_SDMA1_JT                        = 32,   /* SDNA1 JT                 NV      */
	GFX_FW_TYPE_CP_MES                          = 33,   /* CP MES                   NV      */
	GFX_FW_TYPE_MES_STACK                       = 34,   /* MES STACK                NV      */
	GFX_FW_TYPE_RLC_SRM_DRAM_SR                 = 35,   /* RLC SRM DRAM             NV      */
	GFX_FW_TYPE_RLCG_SCRATCH_SR                 = 36,   /* RLCG SCRATCH             NV      */
	GFX_FW_TYPE_RLCP_SCRATCH_SR                 = 37,   /* RLCP SCRATCH             NV      */
	GFX_FW_TYPE_RLCV_SCRATCH_SR                 = 38,   /* RLCV SCRATCH             NV      */
	GFX_FW_TYPE_RLX6_DRAM_SR                    = 39,   /* RLX6 DRAM                NV      */
	GFX_FW_TYPE_SDMA0_PG_CONTEXT                = 40,   /* SDMA0 PG CONTEXT         NV      */
	GFX_FW_TYPE_SDMA1_PG_CONTEXT                = 41,   /* SDMA1 PG CONTEXT         NV      */
	GFX_FW_TYPE_GLOBAL_MUX_SELECT_RAM           = 42,   /* GLOBAL MUX SEL RAM       NV      */
	GFX_FW_TYPE_SE0_MUX_SELECT_RAM              = 43,   /* SE0 MUX SEL RAM          NV      */
	GFX_FW_TYPE_SE1_MUX_SELECT_RAM              = 44,   /* SE1 MUX SEL RAM          NV      */
	GFX_FW_TYPE_ACCUM_CTRL_RAM                  = 45,   /* ACCUM CTRL RAM           NV      */
	GFX_FW_TYPE_RLCP_CAM                        = 46,   /* RLCP CAM                 NV      */
	GFX_FW_TYPE_RLC_SPP_CAM_EXT                 = 47,   /* RLC SPP CAM EXT          NV      */
	GFX_FW_TYPE_RLC_DRAM_BOOT                   = 48,   /* RLC DRAM BOOT            NV      */
	GFX_FW_TYPE_VCN0_RAM                        = 49,   /* VCN_RAM                  NV + RN */
	GFX_FW_TYPE_VCN1_RAM                        = 50,   /* VCN_RAM                  NV + RN */
	GFX_FW_TYPE_DMUB                            = 51,   /* DMUB                          RN */
	GFX_FW_TYPE_SDMA2                           = 52,   /* SDMA2                    MI      */
	GFX_FW_TYPE_SDMA3                           = 53,   /* SDMA3                    MI      */
	GFX_FW_TYPE_SDMA4                           = 54,   /* SDMA4                    MI      */
	GFX_FW_TYPE_SDMA5                           = 55,   /* SDMA5                    MI      */
	GFX_FW_TYPE_SDMA6                           = 56,   /* SDMA6                    MI      */
	GFX_FW_TYPE_SDMA7                           = 57,   /* SDMA7                    MI      */
	GFX_FW_TYPE_VCN1                            = 58,   /* VCN1                     MI      */
	GFX_FW_TYPE_CAP                             = 62,   /* CAP_FW                           */
	GFX_FW_TYPE_SE2_TAP_DELAYS                  = 65,   /* SE2 TAP DELAYS           NV      */
	GFX_FW_TYPE_SE3_TAP_DELAYS                  = 66,   /* SE3 TAP DELAYS           NV      */
	GFX_FW_TYPE_REG_LIST                        = 67,   /* REG_LIST                 MI      */
	GFX_FW_TYPE_IMU_I                           = 68,   /* IMU Instruction FW       SOC21   */
	GFX_FW_TYPE_IMU_D                           = 69,   /* IMU Data FW              SOC21   */
	GFX_FW_TYPE_LSDMA                           = 70,   /* LSDMA FW                 SOC21   */
	GFX_FW_TYPE_SDMA_UCODE_TH0                  = 71,   /* SDMA Thread 0/CTX        SOC21   */
	GFX_FW_TYPE_SDMA_UCODE_TH1                  = 72,   /* SDMA Thread 1/CTL        SOC21   */
	GFX_FW_TYPE_PPTABLE                         = 73,   /* PPTABLE                  SOC21   */
	GFX_FW_TYPE_DISCRETE_USB4                   = 74,   /* dUSB4 FW                 SOC21   */
	GFX_FW_TYPE_TA                              = 75,   /* SRIOV TA FW UUID         SOC21   */
	GFX_FW_TYPE_RS64_MES                        = 76,   /* RS64 MES ucode           SOC21   */
	GFX_FW_TYPE_RS64_MES_STACK                  = 77,   /* RS64 MES stack ucode     SOC21   */
	GFX_FW_TYPE_RS64_KIQ                        = 78,   /* RS64 KIQ ucode           SOC21   */
	GFX_FW_TYPE_RS64_KIQ_STACK                  = 79,   /* RS64 KIQ Heap stack      SOC21   */
	GFX_FW_TYPE_ISP_DATA                        = 80,   /* ISP DATA                 SOC21   */
	GFX_FW_TYPE_CP_MES_KIQ                      = 81,   /* MES KIQ ucode            SOC21   */
	GFX_FW_TYPE_MES_KIQ_STACK                   = 82,   /* MES KIQ stack            SOC21   */
	GFX_FW_TYPE_UMSCH_DATA                      = 83,   /* User Mode Scheduler Data SOC21   */
	GFX_FW_TYPE_UMSCH_UCODE                     = 84,   /* User Mode Scheduler Ucode SOC21  */
	GFX_FW_TYPE_UMSCH_CMD_BUFFER                = 85,   /* User Mode Scheduler Command Buffer SOC21 */
	GFX_FW_TYPE_USB_DP_COMBO_PHY                = 86,   /* USB-Display port Combo   SOC21   */
	GFX_FW_TYPE_RS64_PFP                        = 87,   /* RS64 PFP                 SOC21   */
	GFX_FW_TYPE_RS64_ME                         = 88,   /* RS64 ME                  SOC21   */
	GFX_FW_TYPE_RS64_MEC                        = 89,   /* RS64 MEC                 SOC21   */
	GFX_FW_TYPE_RS64_PFP_P0_STACK               = 90,   /* RS64 PFP stack P0        SOC21   */
	GFX_FW_TYPE_RS64_PFP_P1_STACK               = 91,   /* RS64 PFP stack P1        SOC21   */
	GFX_FW_TYPE_RS64_ME_P0_STACK                = 92,   /* RS64 ME stack P0         SOC21   */
	GFX_FW_TYPE_RS64_ME_P1_STACK                = 93,   /* RS64 ME stack P1         SOC21   */
	GFX_FW_TYPE_RS64_MEC_P0_STACK               = 94,   /* RS64 MEC stack P0        SOC21   */
	GFX_FW_TYPE_RS64_MEC_P1_STACK               = 95,   /* RS64 MEC stack P1        SOC21   */
	GFX_FW_TYPE_RS64_MEC_P2_STACK               = 96,   /* RS64 MEC stack P2        SOC21   */
	GFX_FW_TYPE_RS64_MEC_P3_STACK               = 97,   /* RS64 MEC stack P3        SOC21   */
	GFX_FW_TYPE_VPEC_FW1                        = 100,  /* VPEC FW1 To Save         VPE     */
	GFX_FW_TYPE_VPEC_FW2                        = 101,  /* VPEC FW2 To Save         VPE     */
	GFX_FW_TYPE_VPE                             = 102,
	GFX_FW_TYPE_JPEG_RAM                        = 128,  /**< JPEG Command buffer */
	GFX_FW_TYPE_P2S_TABLE                       = 129,
	GFX_FW_TYPE_MAX
};
