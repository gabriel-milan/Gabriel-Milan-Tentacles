#  Drakkar-Software OctoBot-Interfaces
#  Copyright (c) Drakkar-Software, All rights reserved.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library.

from flask import render_template

from tentacles.Services.Interfaces.web_interface import server_instance, get_logs, flush_errors_count
from tentacles.Services.Interfaces.web_interface.login.web_login_manager import login_required_when_activated


@server_instance.route("/logs")
@login_required_when_activated
def logs():
    flush_errors_count()
    return render_template("logs.html",
                           logs=get_logs())
