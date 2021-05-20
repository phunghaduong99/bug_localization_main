/*******************************************************************************
 * Copyright (c) 2000, 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.jface.tests.dialogs;

import junit.framework.TestCase;

import org.eclipse.jface.dialogs.InputDialog;

public class InputDialogTest extends TestCase {
	
	private InputDialog dialog;
	
	protected void tearDown() throws Exception {
		if (dialog != null) {
			dialog.close();
			dialog = null;
		}
		super.tearDown();
	}

	public void testSetErrorMessageEarly() {
		dialog = new InputDialog(null, "TEST", "value", "test", null);
		dialog.setBlockOnOpen(false);
		dialog.setErrorMessage("error");
		dialog.open();
	}
}
