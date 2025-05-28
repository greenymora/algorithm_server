import sys
import json
from py_iztro import Astro

def format_star_list(star_list):
    return '、'.join([
        f"{star.name}{'('+star.brightness+')' if star.brightness else ''}{'['+star.mutagen+']' if star.mutagen else ''}"
        for star in star_list
    ]) if star_list else '无'

def main():
    if len(sys.argv) != 5:
        print(json.dumps({"error": "参数数量错误"}))
        sys.exit(1)
    gender = sys.argv[1]
    date_type = sys.argv[2]
    date_str = sys.argv[3]
    hour_index = int(sys.argv[4])

    try:
        astro = Astro()
        if date_type == "公历":
            result = astro.by_solar(date_str, hour_index, gender)
        elif date_type == "农历":
            result = astro.by_lunar(date_str, hour_index, gender)
        else:
            print(json.dumps({"error": "date_type 只能为 '公历' 或 '农历'"}))
            sys.exit(1)

        today = result.solar_date
        horoscope = result.horoscope(today)

        lines = []
        lines.append(f"===== 紫微斗数排盘结果（{date_type} {date_str}，{['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥'][hour_index]}时，{gender}）=====")
        lines.append(f"命盘公历生日: {result.solar_date}")
        lines.append(f"命盘农历生日: {result.lunar_date}")
        lines.append(f"四柱: {result.chinese_date}")
        lines.append(f"生肖: {result.zodiac}  星座: {result.sign}")
        lines.append(f"命宫: {result.earthly_branch_of_soul_palace}  身宫: {result.earthly_branch_of_body_palace}")
        lines.append(f"命主: {result.soul}  身主: {result.body}")
        lines.append(f"五行局: {result.five_elements_class}")
        lines.append("")
        for i in range(12):
            palace = result.palaces[i]
            lines.append(
                f"宫位: {palace.name}\n"
                f"  干支: {palace.heavenly_stem}{palace.earthly_branch}\n"
                f"  主星: {format_star_list(palace.major_stars)}\n"
                f"  辅星: {format_star_list(palace.minor_stars)}\n"
                f"  杂曜: {format_star_list(palace.adjective_stars)}\n"
                f"  大运: 大运{horoscope.decadal.palace_names[i]}\n"
                f"    大运星: {format_star_list(horoscope.decadal.stars[i])}\n"
                f"  流年: 流年{horoscope.yearly.palace_names[i]}\n"
                f"    流年星: {format_star_list(horoscope.yearly.stars[i])}\n"
                f"  流月: 流月{horoscope.monthly.palace_names[i]}\n"
                f"    流月星: {format_star_list(horoscope.monthly.stars[i])}\n"
                f"  流日: 流日{horoscope.daily.palace_names[i]}\n"
                f"    流日星: {format_star_list(horoscope.daily.stars[i])}\n"
                f"  流时: 流时{horoscope.hourly.palace_names[i]}\n"
                f"    流时星: {format_star_list(horoscope.hourly.stars[i])}\n"
            )
        print(json.dumps({"result": "\n".join(lines)}, ensure_ascii=False))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))

if __name__ == '__main__':
    main()
