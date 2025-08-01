//
// Copyright (C) 2017 Alexey Khokholov (Nuke.YKT)
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//
//
//  Nuked OPN2(Yamaha YM3438) emulator.
//  Thanks:
//      Silicon Pr0n:
//          Yamaha YM3438 decap and die shot(digshadow).
//      OPLx decapsulated(Matthew Gambrell, Olli Niemitalo):
//          OPL2 ROMs.
//
// version: 1.0.7
//

#ifndef YM3438_H
#define YM3438_H

#include <cstdint>
#define RSM_FRAC 10
#define OPN_WRITEBUF_SIZE 2048
#define OPN_WRITEBUF_DELAY 15

enum {
    ym3438_type_discrete = 0,   /* Discrete YM3438 (Teradrive)          */
    ym3438_type_asic = 1,       /* ASIC YM3438 (MD1 VA7, MD2, MD3, etc) */
    ym3438_type_ym2612 = 2,     /* YM2612 (MD1, MD2 VA2)                */
    ym3438_type_ym2612_u = 3	/* YM2612 without lowpass filter 		*/
};

typedef uint64_t          Bit64u;
typedef int64_t           Bit64s;
typedef uint32_t          Bit32u;
typedef int32_t           Bit32s;
typedef uint16_t          Bit16u;
typedef int16_t           Bit16s;
typedef uint8_t           Bit8u;
typedef int8_t            Bit8s;

typedef struct _opn2_writebuf {
    Bit64u time;
    Bit8u port;
    Bit8u data;
} opn2_writebuf;

typedef struct
{
    Bit32u cycles;
    Bit32u slot;
    Bit32u channel;
    Bit16s mol, mor;
    /* IO */
    Bit16u write_data;
    Bit8u write_a;
    Bit8u write_d;
    Bit8u write_a_en;
    Bit8u write_d_en;
    Bit8u write_busy;
    Bit8u write_busy_cnt;
    Bit8u write_fm_address;
    Bit8u write_fm_data;
    Bit8u write_fm_mode_a;
    Bit16u address;
    Bit8u data;
    Bit8u pin_test_in;
    Bit8u pin_irq;
    Bit8u busy;
    /* LFO */
    Bit8u lfo_en;
    Bit8u lfo_freq;
    Bit8u lfo_pm;
    Bit8u lfo_am;
    Bit8u lfo_cnt;
    Bit8u lfo_inc;
    Bit8u lfo_quotient;
    /* Phase generator */
    Bit16u pg_fnum;
    Bit8u pg_block;
    Bit8u pg_kcode;
    Bit32u pg_inc[24];
    Bit32u pg_phase[24];
    Bit8u pg_reset[24];
    Bit32u pg_read;
    /* Envelope generator */
    Bit8u eg_cycle;
    Bit8u eg_cycle_stop;
    Bit8u eg_shift;
    Bit8u eg_shift_lock;
    Bit8u eg_timer_low_lock;
    Bit16u eg_timer;
    Bit8u eg_timer_inc;
    Bit16u eg_quotient;
    Bit8u eg_custom_timer;
    Bit8u eg_rate;
    Bit8u eg_ksv;
    Bit8u eg_inc;
    Bit8u eg_ratemax;
    Bit8u eg_sl[2];
    Bit8u eg_lfo_am;
    Bit8u eg_tl[2];
    Bit8u eg_state[24];
    Bit16u eg_level[24];
    Bit16u eg_out[24];
    Bit8u eg_kon[24];
    Bit8u eg_kon_csm[24];
    Bit8u eg_kon_latch[24];
    Bit8u eg_csm_mode[24];
    Bit8u eg_ssg_enable[24];
    Bit8u eg_ssg_pgrst_latch[24];
    Bit8u eg_ssg_repeat_latch[24];
    Bit8u eg_ssg_hold_up_latch[24];
    Bit8u eg_ssg_dir[24];
    Bit8u eg_ssg_inv[24];
    Bit32u eg_read[2];
    Bit8u eg_read_inc;
    /* FM */
    Bit16s fm_op1[6][2];
    Bit16s fm_op2[6];
    Bit16s fm_out[24];
    Bit16u fm_mod[24];
    /* Channel */
    Bit16s ch_acc[6];
    Bit16s ch_out[6];
    Bit16s ch_lock;
    Bit8u ch_lock_l;
    Bit8u ch_lock_r;
    Bit16s ch_read;
    /* Timer */
    Bit16u timer_a_cnt;
    Bit16u timer_a_reg;
    Bit8u timer_a_load_lock;
    Bit8u timer_a_load;
    Bit8u timer_a_enable;
    Bit8u timer_a_reset;
    Bit8u timer_a_load_latch;
    Bit8u timer_a_overflow_flag;
    Bit8u timer_a_overflow;
    
    Bit16u timer_b_cnt;
    Bit8u timer_b_subcnt;
    Bit16u timer_b_reg;
    Bit8u timer_b_load_lock;
    Bit8u timer_b_load;
    Bit8u timer_b_enable;
    Bit8u timer_b_reset;
    Bit8u timer_b_load_latch;
    Bit8u timer_b_overflow_flag;
    Bit8u timer_b_overflow;
    
    /* Register set */
    Bit8u mode_test_21[8];
    Bit8u mode_test_2c[8];
    Bit8u mode_ch3;
    Bit8u mode_kon_channel;
    Bit8u mode_kon_operator[4];
    Bit8u mode_kon[24];
    Bit8u mode_csm;
    Bit8u mode_kon_csm;
    Bit8u dacen;
    Bit16s dacdata;
    
    Bit8u ks[24];
    Bit8u ar[24];
    Bit8u sr[24];
    Bit8u dt[24];
    Bit8u multi[24];
    Bit8u sl[24];
    Bit8u rr[24];
    Bit8u dr[24];
    Bit8u am[24];
    Bit8u tl[24];
    Bit8u ssg_eg[24];
    
    Bit16u fnum[6];
    Bit8u block[6];
    Bit8u kcode[6];
    Bit16u fnum_3ch[6];
    Bit8u block_3ch[6];
    Bit8u kcode_3ch[6];
    Bit8u reg_a4;
    Bit8u reg_ac;
    Bit8u connect[6];
    Bit8u fb[6];
    Bit8u pan_l[6], pan_r[6];
    Bit8u ams[6];
    Bit8u pms[6];

	Bit32u mute[7];
	Bit32s rateratio;
	Bit32s samplecnt;
	Bit32s oldsamples[2];
	Bit32s samples[2];

    Bit64u writebuf_samplecnt;
    Bit32u writebuf_cur;
    Bit32u writebuf_last;
    Bit64u writebuf_lasttime;
    opn2_writebuf writebuf[OPN_WRITEBUF_SIZE];
} ym3438_t;

void OPN2_Reset(ym3438_t *chip, Bit32u rate, Bit32u clock);
void OPN2_SetChipType(Bit32u type);
void OPN2_Clock(ym3438_t *chip, Bit32s *buffer);
void OPN2_Write(ym3438_t *chip, Bit32u port, Bit8u data);
void OPN2_SetTestPin(ym3438_t *chip, Bit32u value);
Bit32u OPN2_ReadTestPin(ym3438_t *chip);
Bit32u OPN2_ReadIRQPin(ym3438_t *chip);
Bit8u OPN2_Read(ym3438_t *chip, Bit32u port);
void OPN2_WriteBuffered(ym3438_t *chip, Bit32u port, Bit8u data);
void OPN2_GenerateStream(ym3438_t *chip, Bit32s **sndptr, Bit32u numsamples);
void OPN2_SetOptions(Bit8u flags);
void OPN2_SetMute(ym3438_t *chip, Bit32u mute);
#endif
