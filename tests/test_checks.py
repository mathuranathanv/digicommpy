"""Release gate. Each test is one of the chapter checks already in the package."""

from digicommpy.chapter1.check_rest import main as check_chapter1
from digicommpy.chapter2.check_modems import main as check_modems
from digicommpy.chapter4.check_performance import main as check_performance
from digicommpy.chapter5.check_sync import main as check_sync
from digicommpy.chapter6.check_rest import (
    check_coding,
    check_equalize,
    check_impair,
    check_mimo,
    check_ofdm,
)


def test_chapter1() -> None:
    check_chapter1()


def test_chapters_2_and_3() -> None:
    check_modems()


def test_chapter4() -> None:
    check_performance()


def test_chapter5() -> None:
    check_sync()


def test_chapter6() -> None:
    check_equalize()


def test_chapter7() -> None:
    check_impair()


def test_chapter8() -> None:
    check_ofdm()


def test_chapter9() -> None:
    check_mimo()


def test_chapter10() -> None:
    check_coding()
